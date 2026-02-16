"""Alpamayo R1 model-based controller for CARLA ego vehicle."""

from typing import Any

import carla
import cv2
import numpy as np
import scipy.spatial.transform as spt
import torch
from einops import rearrange
from PIL import Image

from alpamayo_r1 import helper


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
        control_frequency: float = 10.0,
        save_video: bool = True,
        video_path: str = "trajectory_visualization.mp4",
    ):
        """Initialize Alpamayo controller.

        Args:
            ego_vehicle: CARLA ego vehicle actor
            cameras: Dictionary mapping camera names to camera actors
            model: Alpamayo R1 model instance (optional)
            processor: Model processor/tokenizer instance (optional)
            control_frequency: Control update frequency in Hz (default: 10Hz)
            save_video: Whether to save visualization video (default: True)
            video_path: Path to save video (default: trajectory_visualization.mp4)
        """
        self.ego_vehicle = ego_vehicle
        self.cameras = cameras
        self.model = model
        self.processor = processor
        self.control_frequency = control_frequency
        self.save_video = save_video
        self.video_path = video_path

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
        self.world_snapshot = None

        # Pre-allocated tensors for padding (VRAM optimization)
        self._cached_padding_tensors = {
            'xyz': {},  # Cache by pad_length
            'rot': {},  # Cache by pad_length
        }

        # Video writer for trajectory visualization
        self.video_writer = None
        if self.save_video:
            self._init_video_writer()

        # Camera parameters for projection
        self._init_camera_params()

        # Initialize vehicle control settings
        self._init_vehicle_control()

    def _init_vehicle_control(self) -> None:
        """Initialize vehicle control settings (gear, handbrake, etc.)."""
        # Disable autopilot to ensure manual control
        self.ego_vehicle.set_autopilot(False)
        print("Autopilot disabled for ego vehicle")

        # Create initial control to set up vehicle
        initial_control = carla.VehicleControl()
        initial_control.manual_gear_shift = False  # Use automatic transmission
        initial_control.hand_brake = False  # Release handbrake
        initial_control.gear = 1  # Set to first gear (forward)
        initial_control.throttle = 0.0
        initial_control.steer = 0.0
        initial_control.brake = 0.0

        # Apply initial control
        self.ego_vehicle.apply_control(initial_control)
        print("Vehicle control initialized: automatic transmission, handbrake released")

    def _init_video_writer(self) -> None:
        """Initialize video writer for trajectory visualization."""
        # Video parameters (1920x1080 at 20 FPS to match CARLA simulation)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            self.video_path,
            fourcc,
            20.0,  # FPS (CARLA default)
            (1920, 1080)  # Resolution
        )
        print(f"Initialized video writer: {self.video_path}")

    def _init_camera_params(self) -> None:
        """Initialize camera parameters for projection."""
        # Get front wide camera
        front_camera = self.cameras.get("camera_front_wide_120fov")
        if front_camera is None:
            print("Warning: camera_front_wide_120fov not found")
            return

        # Camera intrinsics (1920x1080, 120° FOV)
        self.image_width = 1920
        self.image_height = 1080
        self.fov = 120.0  # degrees

        # Calculate focal length from FOV
        # f = (image_width / 2) / tan(fov / 2)
        fov_rad = np.radians(self.fov)
        self.focal_length = (self.image_width / 2.0) / np.tan(fov_rad / 2.0)

        # Camera intrinsic matrix
        cx = self.image_width / 2.0
        cy = self.image_height / 2.0
        self.K = np.array([
            [self.focal_length, 0, cx],
            [0, self.focal_length, cy],
            [0, 0, 1]
        ])

        # Camera extrinsics (relative to ego vehicle)
        # Get relative transform from camera sensor
        camera_transform = front_camera.get_transform()

        # Camera position relative to ego vehicle (in ego vehicle frame)
        # CARLA camera is typically mounted at [1.5, 0, 2.0] for front camera
        cam_loc = camera_transform.location
        self.camera_offset = np.array([cam_loc.x, cam_loc.y, cam_loc.z])

        # Camera rotation relative to ego vehicle
        cam_rot = camera_transform.rotation
        pitch = np.radians(cam_rot.pitch)
        yaw = np.radians(cam_rot.yaw)
        roll = np.radians(cam_rot.roll)

        # Store relative rotation (this doesn't change as camera is fixed to vehicle)
        self.camera_rotation = spt.Rotation.from_euler('zyx', [yaw, pitch, roll])

    def _project_trajectory_to_image(self, trajectory_local: np.ndarray) -> tuple:
        """Project 3D trajectory to 2D image coordinates.

        Args:
            trajectory_local: Trajectory in ego vehicle local frame (N, 3)
                             where X=forward, Y=left, Z=up

        Returns:
            Tuple of (2D image coordinates (N, 2), valid mask (N,))
        """
        if trajectory_local.shape[0] == 0:
            return np.array([]), np.array([])

        # Transform from ego vehicle frame to camera frame
        # Ego frame: X=forward, Y=left, Z=up
        # Camera frame (CARLA): X=forward, Y=right, Z=up
        # Note: Camera rotation is usually identity for front camera

        # Translate to camera position (camera is offset from ego center)
        traj_translated = trajectory_local - self.camera_offset

        # Apply camera rotation (if camera is not aligned with vehicle)
        # This converts from ego frame to camera frame
        traj_cam = self.camera_rotation.inv().apply(traj_translated)

        # Filter points behind camera (X > 0 in CARLA camera frame)
        # In CARLA camera coordinate: X=forward, Y=right, Z=up
        valid_mask = traj_cam[:, 0] > 0.1  # At least 10cm in front

        # Convert to standard computer vision camera frame
        # CARLA camera: X=forward, Y=right, Z=up
        # Standard CV: X=right, Y=down, Z=forward
        # CARLA: [X, Y, Z] -> Standard CV: [Y, -Z, X]
        traj_cam_cv = np.zeros_like(traj_cam)
        traj_cam_cv[:, 0] = traj_cam[:, 1]   # Y_carla -> X_cv (right)
        traj_cam_cv[:, 1] = -traj_cam[:, 2]  # -Z_carla -> Y_cv (down)
        traj_cam_cv[:, 2] = traj_cam[:, 0]   # X_carla -> Z_cv (forward/depth)

        # Project using pinhole camera model
        # [u, v, 1]^T = (1/Z) * K * [X, Y, Z]^T
        points_2d = np.zeros((traj_cam_cv.shape[0], 2))

        for i in range(traj_cam_cv.shape[0]):
            if valid_mask[i] and traj_cam_cv[i, 2] > 0:
                point_3d = traj_cam_cv[i]
                point_2d_homo = self.K @ point_3d
                points_2d[i] = point_2d_homo[:2] / point_2d_homo[2]
            else:
                points_2d[i] = [-1, -1]  # Invalid point marker

        return points_2d, valid_mask

    def _draw_trajectory_on_image(self, image: np.ndarray, trajectory_local: np.ndarray) -> np.ndarray:
        """Draw predicted trajectory on camera image.

        Args:
            image: Camera image (H, W, 3) in RGB format
            trajectory_local: Trajectory in ego vehicle local frame (N, 3)

        Returns:
            Image with trajectory drawn (H, W, 3) in RGB format
        """
        if trajectory_local is None or len(trajectory_local) == 0:
            return image

        # Convert RGB to BGR for OpenCV
        img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Project trajectory to image
        points_2d, valid_mask = self._project_trajectory_to_image(trajectory_local)

        if points_2d.shape[0] == 0:
            return image

        # Draw trajectory as connected line segments
        prev_point = None
        for i, (point, valid) in enumerate(zip(points_2d, valid_mask)):
            if not valid:
                prev_point = None
                continue

            u, v = int(point[0]), int(point[1])

            # Check if point is within image bounds
            if 0 <= u < self.image_width and 0 <= v < self.image_height:
                # Draw point
                cv2.circle(img_bgr, (u, v), 3, (0, 255, 0), -1)  # Green circle

                # Draw line from previous point
                if prev_point is not None:
                    prev_u, prev_v = int(prev_point[0]), int(prev_point[1])
                    if 0 <= prev_u < self.image_width and 0 <= prev_v < self.image_height:
                        cv2.line(img_bgr, (prev_u, prev_v), (u, v), (0, 255, 0), 2)  # Green line

                prev_point = point
            else:
                prev_point = None

        # Convert back to RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return img_rgb

    def update(self, world_snapshot: Any, camera_images: dict[str, np.ndarray]) -> None:
        """Update controller state and compute control commands.

        Args:
            world_snapshot: CARLA world snapshot
            camera_images: Dictionary mapping camera names to RGB images (H, W, 3)
        """
        self.step_count += 1
        self.world_snapshot = world_snapshot

        # Update ego state
        self._update_ego_state()

        # Run model inference at control frequency (10Hz)
        # Only store images when inference is needed to save memory
        if self.step_count - self.last_control_step >= (1.0 / self.control_frequency) * 20:
            self.current_images = camera_images  # Only store when needed
            self._run_inference()
            self.last_control_step = self.step_count

        # Apply control EVERY frame (using latest prediction from model)
        # This allows smooth control at simulation rate (20Hz) while inference runs at 10Hz
        self._apply_control()

        # Save visualization video
        if self.save_video and self.video_writer is not None:
            self._save_visualization_frame(camera_images)

    def _save_visualization_frame(self, camera_images: dict[str, np.ndarray]) -> None:
        """Save current frame with trajectory visualization to video.

        Args:
            camera_images: Dictionary mapping camera names to RGB images
        """
        # Get front wide camera image
        front_image = camera_images.get("camera_front_wide_120fov")
        if front_image is None:
            return

        # Draw trajectory on image if available
        if self.predicted_trajectory is not None and len(self.predicted_trajectory) > 0:
            visualized_image = self._draw_trajectory_on_image(
                front_image,
                self.predicted_trajectory
            )
        else:
            visualized_image = front_image

        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(visualized_image, cv2.COLOR_RGB2BGR)

        # Write frame to video
        self.video_writer.write(frame_bgr)

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
        if self.model is None or self.processor is None or len(self.current_images) == 0:
            return

        try:
            # Prepare input data
            model_input = self._prepare_model_input()

            # Run inference with inference_mode (more efficient than no_grad) and autocast
            # inference_mode disables view tracking and version counter, reducing memory overhead
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                pred_xyz, pred_rot, extra = self.model.sample_trajectories_from_data_with_vlm_rollout(
                    data=model_input,
                    top_p=0.98,
                    temperature=0.6,
                    num_traj_samples=1,
                    max_generation_length=256,  # Same as test_inference.py
                    return_extra=True,
                )

            # Extract predicted trajectory (use first sample)
            # pred_xyz shape: [batch_size, num_traj_sets, num_traj_samples, num_timesteps, 3]
            self.predicted_trajectory = pred_xyz.cpu().numpy()[0, 0, 0]  # (num_timesteps, 3)

            # Delete intermediate tensors to free memory immediately
            del pred_xyz, pred_rot, extra, model_input

            # Clear CUDA cache to free memory for next inference
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"Model inference failed: {e}")
            import traceback
            traceback.print_exc()

            # Aggressive memory cleanup on error
            if 'model_input' in locals():
                del model_input
            torch.cuda.empty_cache()

            # Exit on model error (especially CUDA OOM)
            print("Exiting due to model inference failure.")
            print("If you're still getting CUDA OOM, consider:")
            print("  1. Further reducing image resolution in controller.py")
            print("  2. Reducing max_generation_length (currently 256)")
            print("  3. Ensuring Flash Attention 2 is properly installed")
            import sys
            sys.exit(1)

    def _prepare_model_input(self) -> dict[str, Any]:
        """Prepare input tensors for model inference.

        Returns:
            Dictionary of input tensors for the model
        """
        # Use only the 4 cameras that test_inference.py uses (to match memory usage)
        # Indices: 0, 1, 2, 6
        used_cameras = [
            "camera_cross_left_120fov",      # index 0
            "camera_front_wide_120fov",      # index 1
            "camera_cross_right_120fov",     # index 2
            "camera_front_tele_30fov",       # index 6
        ]

        camera_name_to_index = {
            "camera_cross_left_120fov": 0,
            "camera_front_wide_120fov": 1,
            "camera_cross_right_120fov": 2,
            "camera_rear_left_70fov": 3,
            "camera_rear_tele_30fov": 4,
            "camera_rear_right_70fov": 5,
            "camera_front_tele_30fov": 6,
        }

        # Filter and sort cameras by index (only use the 4 cameras)
        sorted_cameras = sorted(
            [(name, img) for name, img in self.current_images.items() if name in used_cameras],
            key=lambda x: camera_name_to_index.get(x[0], 999)
        )

        image_list = []
        for cam_name, image in sorted_cameras:
            # Use native resolution (same as test_inference.py)
            # CARLA provides 1920x1080 images
            # Convert to tensor (H, W, C) -> (C, H, W)
            # Copy the array to ensure it's writable and has positive strides
            img_tensor = torch.from_numpy(image.copy()).float()
            img_tensor = rearrange(img_tensor, "h w c -> c h w")
            image_list.append(img_tensor)

        # Stack images: (N_cameras, C, H, W)
        images = torch.stack(image_list, dim=0)

        # Create messages from images using helper
        # helper.create_message expects (N, C, H, W)
        messages = helper.create_message(images)

        # Tokenize using processor
        tokenized_inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            continue_final_message=True,
            return_dict=True,
            return_tensors="pt",
        )

        # Prepare ego history
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

            # Add batch and temporal dimensions to match expected shape
            ego_history_xyz = torch.from_numpy(history_xyz_local).float()
            ego_history_xyz = ego_history_xyz.unsqueeze(0).unsqueeze(0)  # (1, 1, T, 3)

            ego_history_rot = torch.from_numpy(history_rot_local).float()
            ego_history_rot = ego_history_rot.unsqueeze(0).unsqueeze(0)  # (1, 1, T, 3, 3)

            # Pad to 16 timesteps if needed (using cached tensors to save memory)
            if num_history < 16:
                pad_length = 16 - num_history

                # Use cached padding tensors
                if pad_length not in self._cached_padding_tensors['xyz']:
                    self._cached_padding_tensors['xyz'][pad_length] = torch.zeros(1, 1, pad_length, 3)
                if pad_length not in self._cached_padding_tensors['rot']:
                    self._cached_padding_tensors['rot'][pad_length] = torch.eye(3).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(1, 1, pad_length, 1, 1)

                ego_history_xyz = torch.cat([
                    self._cached_padding_tensors['xyz'][pad_length],
                    ego_history_xyz
                ], dim=2)
                ego_history_rot = torch.cat([
                    self._cached_padding_tensors['rot'][pad_length],
                    ego_history_rot
                ], dim=2)
        else:
            # Use zeros if no history available (cached)
            if 16 not in self._cached_padding_tensors['xyz']:
                self._cached_padding_tensors['xyz'][16] = torch.zeros(1, 1, 16, 3)
            if 16 not in self._cached_padding_tensors['rot']:
                self._cached_padding_tensors['rot'][16] = torch.eye(3).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(1, 1, 16, 1, 1)

            ego_history_xyz = self._cached_padding_tensors['xyz'][16]
            ego_history_rot = self._cached_padding_tensors['rot'][16]

        # Prepare model inputs
        model_inputs = {
            "tokenized_data": tokenized_inputs,
            "ego_history_xyz": ego_history_xyz,
            "ego_history_rot": ego_history_rot,
        }

        # Move all tensors to CUDA
        model_inputs = helper.to_device(model_inputs, "cuda")

        return model_inputs

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
        control.manual_gear_shift = False  # Automatic transmission
        control.hand_brake = False  # Ensure handbrake is off
        control.throttle = float(throttle)
        control.steer = float(steering)
        control.brake = float(brake)

        # Get vehicle state for debugging
        vehicle_transform = self.ego_vehicle.get_transform()
        vehicle_location = vehicle_transform.location
        vehicle_control_state = self.ego_vehicle.get_control()

        # Debug output with detailed vehicle state
        print(f"[Control] Step: {self.step_count:4d} | "
              f"Target: ({target_x:5.2f}, {target_y:5.2f}) | "
              f"Speed: {current_speed:4.1f}/{self.target_speed:4.1f} m/s | "
              f"Vel: ({current_velocity.x:5.2f}, {current_velocity.y:5.2f}, {current_velocity.z:5.2f}) | "
              f"Pos: ({vehicle_location.x:7.2f}, {vehicle_location.y:7.2f}) | "
              f"Throttle: {control.throttle:.3f} | "
              f"Steer: {control.steer:6.3f} | "
              f"Brake: {control.brake:.3f} | "
              f"Gear: {vehicle_control_state.gear}")

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
        control.manual_gear_shift = False  # Automatic transmission
        control.hand_brake = False  # Ensure handbrake is off
        control.throttle = float(throttle)
        control.steer = 0.0
        control.brake = 0.0

        # Debug output
        print(f"[Control-Simple] Step: {self.step_count:4d} | "
              f"Speed: {current_speed:4.1f}/{self.target_speed:4.1f} m/s | "
              f"Throttle: {control.throttle:.3f} | "
              f"(No trajectory available)")

        self.ego_vehicle.apply_control(control)

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

    def close(self) -> None:
        """Clean up resources (close video writer)."""
        if self.video_writer is not None:
            self.video_writer.release()
            print(f"Video saved to: {self.video_path}")
            self.video_writer = None

    def __del__(self) -> None:
        """Destructor to ensure video writer is closed."""
        self.close()
