"""Alpamayo R1 model-based controller for CARLA ego vehicle."""

from typing import Any

import carla
import numpy as np
import scipy.spatial.transform as spt
import torch
from einops import rearrange


class AlpamayoController:
    """Controller for ego vehicle using Alpamayo R1 model predictions.

    This class manages:
    - Camera image collection from CARLA sensors
    - Ego vehicle state tracking (position, rotation, velocity)
    - Model inference with Alpamayo R1
    - Vehicle control command generation from predicted trajectories
    - Optional Rerun visualization
    """

    def __init__(
        self,
        ego_vehicle: carla.Actor,
        cameras: dict[str, carla.Actor],
        model: Any = None,
        processor: Any = None,
        use_rerun: bool = False,
        control_frequency: float = 10.0,
    ):
        """Initialize Alpamayo controller.

        Args:
            ego_vehicle: CARLA ego vehicle actor
            cameras: Dictionary mapping camera names to camera actors
            model: Alpamayo R1 model instance (optional)
            processor: Model processor/tokenizer instance (optional)
            use_rerun: Whether to enable Rerun visualization
            control_frequency: Control update frequency in Hz (default: 10Hz)
        """
        self.ego_vehicle = ego_vehicle
        self.cameras = cameras
        self.model = model
        self.processor = processor
        self.use_rerun = use_rerun
        self.control_frequency = control_frequency

        # Control parameters
        self.target_speed = 5.0  # m/s (about 18 km/h)
        self.max_speed = 15.0  # m/s (about 54 km/h)
        self.max_steering = 0.8  # radians

        # State tracking
        self.ego_history_xyz = []
        self.ego_history_rot = []
        self.step_count = 0
        self.last_control_step = 0

        # Latest predictions
        self.predicted_trajectory = None
        self.current_images = {}

        # Initialize Rerun if enabled
        if self.use_rerun:
            self._init_rerun()

    def _init_rerun(self) -> None:
        """Initialize Rerun for visualization."""
        try:
            import rerun as rr

            rr.init("alpamayo_carla", spawn=True)

            # Set up coordinate system
            rr.log(
                "world",
                rr.ViewCoordinates.RIGHT_HAND_Y_DOWN,
                static=True,
            )

            print("Rerun visualization initialized")
        except ImportError:
            print("Warning: rerun-sdk not installed, visualization disabled")
            self.use_rerun = False

    def update(self, world_snapshot: Any, camera_images: dict[str, np.ndarray]) -> None:
        """Update controller state and compute control commands.

        Args:
            world_snapshot: CARLA world snapshot
            camera_images: Dictionary mapping camera names to RGB images (H, W, 3)
        """
        self.step_count += 1
        self.current_images = camera_images

        # Update ego state
        self._update_ego_state()

        # Run model inference at control frequency
        if self.step_count - self.last_control_step >= (1.0 / self.control_frequency) * 20:
            self._run_inference()
            self._apply_control()
            self.last_control_step = self.step_count

        # Update visualization
        if self.use_rerun:
            self._update_rerun()

    def _update_ego_state(self) -> None:
        """Update ego vehicle state history."""
        transform = self.ego_vehicle.get_transform()
        location = transform.location
        rotation = transform.rotation

        # Convert to numpy arrays
        xyz = np.array([location.x, location.y, location.z])

        # Convert rotation to rotation matrix
        # CARLA uses pitch, yaw, roll in degrees
        pitch_rad = np.radians(rotation.pitch)
        yaw_rad = np.radians(rotation.yaw)
        roll_rad = np.radians(rotation.roll)

        # Create rotation matrix from Euler angles (ZYX order)
        rot = spt.Rotation.from_euler("zyx", [yaw_rad, pitch_rad, roll_rad])
        rot_matrix = rot.as_matrix()

        # Store history
        self.ego_history_xyz.append(xyz)
        self.ego_history_rot.append(rot_matrix)

        # Keep only recent history (e.g., last 2 seconds at 20Hz = 40 steps)
        max_history = 40
        if len(self.ego_history_xyz) > max_history:
            self.ego_history_xyz = self.ego_history_xyz[-max_history:]
            self.ego_history_rot = self.ego_history_rot[-max_history:]

    def _run_inference(self) -> None:
        """Run Alpamayo R1 model inference on current observations."""
        if self.model is None or len(self.current_images) == 0:
            return

        try:
            # Prepare input data
            model_input = self._prepare_model_input()

            # Run inference
            with torch.no_grad():
                outputs = self.model(**model_input)

            # Extract predicted trajectory
            self.predicted_trajectory = self._extract_trajectory(outputs)

        except Exception as e:
            print(f"Model inference failed: {e}")
            self.predicted_trajectory = None

    def _prepare_model_input(self) -> dict[str, torch.Tensor]:
        """Prepare input tensors for model inference.

        Returns:
            Dictionary of input tensors for the model
        """
        # Convert camera images to tensors
        image_list = []
        camera_indices = []

        camera_name_to_index = {
            "camera_cross_left_120fov": 0,
            "camera_front_wide_120fov": 1,
            "camera_cross_right_120fov": 2,
            "camera_rear_left_70fov": 3,
            "camera_rear_tele_30fov": 4,
            "camera_rear_right_70fov": 5,
            "camera_front_tele_30fov": 6,
        }

        for cam_name, image in sorted(self.current_images.items()):
            # Convert to tensor and normalize
            # Use .copy() to ensure contiguous memory layout (fixes negative stride issue)
            img_tensor = torch.from_numpy(image.copy()).float() / 255.0
            img_tensor = rearrange(img_tensor, "h w c -> c h w")
            image_list.append(img_tensor)

            cam_idx = camera_name_to_index.get(cam_name, 0)
            camera_indices.append(cam_idx)

        # Stack images: (N_cameras, 3, H, W)
        images = torch.stack(image_list, dim=0)

        # Add batch and frame dimensions: (1, N_cameras, 1, 3, H, W)
        images = images.unsqueeze(0).unsqueeze(2)

        # Prepare ego history (simplified - use recent history)
        num_history = min(16, len(self.ego_history_xyz))
        if num_history > 0:
            history_xyz = np.array(self.ego_history_xyz[-num_history:])
            history_rot = np.array(self.ego_history_rot[-num_history:])

            # Transform to local frame (relative to current pose)
            current_xyz = history_xyz[-1]
            current_rot = spt.Rotation.from_matrix(history_rot[-1])
            current_rot_inv = current_rot.inv()

            history_xyz_local = current_rot_inv.apply(history_xyz - current_xyz)
            history_rot_local = (
                current_rot_inv * spt.Rotation.from_matrix(history_rot)
            ).as_matrix()

            ego_history_xyz = torch.from_numpy(history_xyz_local).float().unsqueeze(0).unsqueeze(0)
            ego_history_rot = torch.from_numpy(history_rot_local).float().unsqueeze(0).unsqueeze(0)
        else:
            # Use zeros if no history available
            ego_history_xyz = torch.zeros(1, 1, 16, 3)
            ego_history_rot = torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(1, 1, 16, 1, 1)

        return {
            "image_frames": images,
            "camera_indices": torch.tensor(camera_indices),
            "ego_history_xyz": ego_history_xyz,
            "ego_history_rot": ego_history_rot,
        }

    def _extract_trajectory(self, outputs: Any) -> np.ndarray:
        """Extract trajectory from model outputs.

        Args:
            outputs: Model output dictionary

        Returns:
            Predicted trajectory as numpy array of shape (T, 3) in local frame
        """
        # This is a placeholder - actual implementation depends on model output format
        # Assuming outputs contain 'predicted_trajectory' key
        if hasattr(outputs, "predicted_trajectory"):
            traj = outputs.predicted_trajectory
            if isinstance(traj, torch.Tensor):
                traj = traj.cpu().numpy()
            return traj[0, 0]  # (T, 3)

        # Fallback: return straight line trajectory
        T = 64
        trajectory = np.zeros((T, 3))
        trajectory[:, 0] = np.linspace(0, 10, T)  # 10m forward
        return trajectory

    def _apply_control(self) -> None:
        """Apply control commands based on predicted trajectory."""
        if self.predicted_trajectory is None or len(self.predicted_trajectory) == 0:
            # Fallback: maintain current speed
            self._apply_simple_control()
            return

        # Get near-term target point (e.g., 1 second ahead at 10Hz = 10th point)
        lookahead_idx = min(10, len(self.predicted_trajectory) - 1)
        target_point = self.predicted_trajectory[lookahead_idx]

        # Calculate steering angle using pure pursuit
        target_x = target_point[0]
        target_y = target_point[1]

        # Lateral error
        lateral_error = target_y

        # Calculate steering angle (simple proportional control)
        steering_gain = 0.5
        steering = np.clip(steering_gain * lateral_error, -self.max_steering, self.max_steering)

        # Calculate throttle based on desired speed
        current_velocity = self.ego_vehicle.get_velocity()
        current_speed = np.sqrt(
            current_velocity.x**2 + current_velocity.y**2 + current_velocity.z**2
        )

        speed_error = self.target_speed - current_speed
        throttle = np.clip(0.5 * speed_error, 0.0, 1.0)
        brake = 0.0 if speed_error > -0.5 else 0.3

        # Apply control
        control = carla.VehicleControl()
        control.throttle = float(throttle)
        control.steer = float(steering)
        control.brake = float(brake)

        self.ego_vehicle.apply_control(control)

    def _apply_simple_control(self) -> None:
        """Apply simple speed control when no trajectory available."""
        current_velocity = self.ego_vehicle.get_velocity()
        current_speed = np.sqrt(
            current_velocity.x**2 + current_velocity.y**2 + current_velocity.z**2
        )

        speed_error = self.target_speed - current_speed
        throttle = np.clip(0.5 * speed_error, 0.0, 1.0)

        control = carla.VehicleControl()
        control.throttle = float(throttle)
        control.steer = 0.0
        control.brake = 0.0

        self.ego_vehicle.apply_control(control)

    def _update_rerun(self) -> None:
        """Update Rerun visualization."""
        try:
            import rerun as rr

            # Log ego vehicle position
            if len(self.ego_history_xyz) > 0:
                current_pos = self.ego_history_xyz[-1]
                rr.log(
                    "world/ego_vehicle",
                    rr.Points3D([current_pos], colors=[[0, 255, 0]], radii=[0.5]),
                )

                # Log ego trajectory history
                if len(self.ego_history_xyz) > 1:
                    history_array = np.array(self.ego_history_xyz)
                    rr.log(
                        "world/ego_trajectory/history",
                        rr.LineStrips3D([history_array], colors=[[0, 200, 0]]),
                    )

            # Log predicted trajectory
            if self.predicted_trajectory is not None and len(self.predicted_trajectory) > 0:
                # Transform predicted trajectory to world frame
                current_xyz = self.ego_history_xyz[-1]
                current_rot = spt.Rotation.from_matrix(self.ego_history_rot[-1])

                # Transform local predictions to world coordinates
                pred_world = current_rot.apply(self.predicted_trajectory[:, :3]) + current_xyz

                rr.log(
                    "world/ego_trajectory/prediction",
                    rr.LineStrips3D([pred_world], colors=[[255, 0, 0]]),
                )

            # Log camera images
            for cam_name, image in self.current_images.items():
                rr.log(f"cameras/{cam_name}", rr.Image(image))

        except Exception as e:
            print(f"Rerun visualization update failed: {e}")

    def set_target_speed(self, speed: float) -> None:
        """Set target speed for the controller.

        Args:
            speed: Target speed in m/s
        """
        self.target_speed = np.clip(speed, 0.0, self.max_speed)

    def get_current_speed(self) -> float:
        """Get current ego vehicle speed.

        Returns:
            Current speed in m/s
        """
        velocity = self.ego_vehicle.get_velocity()
        return np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)

    def get_current_position(self) -> np.ndarray:
        """Get current ego vehicle position.

        Returns:
            Position as numpy array [x, y, z]
        """
        if len(self.ego_history_xyz) > 0:
            return self.ego_history_xyz[-1].copy()
        else:
            transform = self.ego_vehicle.get_transform()
            location = transform.location
            return np.array([location.x, location.y, location.z])
