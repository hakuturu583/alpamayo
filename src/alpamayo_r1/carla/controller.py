"""Alpamayo R1 model-based controller for CARLA ego vehicle."""

from collections import deque
from typing import Any

import carla
import cv2
import numpy as np
import scipy.spatial.transform as spt
import torch
from einops import rearrange
from PIL import Image

from alpamayo_r1 import helper
from alpamayo_r1.carla.config import CarlaConfig
from alpamayo_r1.carla.decoupled_controller import DecoupledController


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
        config: CarlaConfig | None = None,
        save_video: bool = True,
        log_dir: str = None,
    ):
        """Initialize Alpamayo controller.

        Args:
            ego_vehicle: CARLA ego vehicle actor
            cameras: Dictionary mapping camera names to camera actors
            model: Alpamayo R1 model instance (optional)
            processor: Model processor/tokenizer instance (optional)
            config: Full pipeline configuration (defaults to CarlaConfig())
            save_video: Whether to save visualization video (default: True)
            log_dir: Directory to save video logs (default: log/YYYYMMDD_HHMMSS)
        """
        self._config = config or CarlaConfig()
        self.ego_vehicle = ego_vehicle
        self.cameras = cameras
        self.model = model
        self.processor = processor
        self.control_frequency = self._config.inference.control_frequency
        self.save_video = save_video

        # Create log directory with timestamp
        if log_dir is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_dir = f"log/{timestamp}"
        self.log_dir = log_dir

        if self.save_video:
            import os
            os.makedirs(self.log_dir, exist_ok=True)
            print(f"Video logs will be saved to: {self.log_dir}")

        # Control parameters
        ctrl = self._config.control
        self.max_speed = ctrl.max_speed
        self.min_lookahead_distance = ctrl.min_lookahead_distance
        self.lookahead_time = 2.0  # seconds (legacy pure pursuit parameter)
        # wheelbase and max_steering are read from CARLA vehicle physics in
        # _log_vehicle_physics(); accessing them before that call raises AttributeError.

        # State tracking
        self.ego_history_xyz = []
        self.ego_history_rot = []
        self.step_count = 0
        self.last_control_step = 0

        # Latest predictions
        self.predicted_trajectory = None
        self.current_images = {}
        self.world_snapshot = None

        # Rolling camera frame buffer: stores last NUM_FRAME_HISTORY frames per camera
        # Matches training data: [t0-0.3s, t0-0.2s, t0-0.1s, t0] at 10Hz
        self.NUM_FRAME_HISTORY = self._config.inference.num_frame_history
        self.camera_frame_buffer: dict[str, deque] = {}

        # Language traces (CoT, meta action, answer)
        self.latest_cot = ""
        self.latest_meta_action = ""
        self.latest_answer = ""

        # Decoupled controller (initialized after vehicle physics are loaded)
        self.decoupled_controller = None

        # Pre-allocated tensors for padding (VRAM optimization)
        self._cached_padding_tensors = {
            'xyz': {},  # Cache by pad_length
            'rot': {},  # Cache by pad_length
        }

        # Video writer for trajectory visualization
        self.video_writer = None
        self.video_frame_count = 0
        self.video_segment_index = 0
        self.frames_per_segment = self._config.video.frames_per_segment
        if self.save_video:
            self._init_video_writer()

        # Camera parameters for projection
        self._init_camera_params()

        # Initialize vehicle control settings
        self._init_vehicle_control()

        # Log vehicle physics parameters for comparison
        self._log_vehicle_physics()

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

    def _log_vehicle_physics(self) -> None:
        """Log vehicle physics parameters and compare with controller settings."""
        print("\n" + "="*80)
        print("VEHICLE PHYSICS PARAMETER COMPARISON")
        print("="*80)

        # Get CARLA vehicle physics
        physics = self.ego_vehicle.get_physics_control()

        # CARLA stores 4 wheels: [front_left, front_right, rear_left, rear_right]
        wheels = physics.wheels
        if len(wheels) < 4:
            raise RuntimeError(
                f"Cannot read vehicle parameters from CARLA physics: "
                f"expected 4 wheels, got {len(wheels)}"
            )

        # Extract wheelbase from wheel positions (CARLA stores positions in cm)
        front_wheel = wheels[0]  # front left
        rear_wheel = wheels[2]   # rear left
        self.wheelbase = abs(front_wheel.position.x - rear_wheel.position.x) / 100.0  # cm → m
        print(f"\n[Wheelbase]")
        print(f"  CARLA actual:        {self.wheelbase:.3f} m")

        # Extract max steering angle from front wheel
        max_steer_angle_deg = wheels[0].max_steer_angle  # degrees
        self.max_steering = np.deg2rad(max_steer_angle_deg)
        print(f"\n[Max Steering Angle]")
        print(f"  CARLA actual:        {self.max_steering:.3f} rad ({max_steer_angle_deg:.1f} deg)")

        # Max curvature achievable with actual steering
        max_curvature = np.tan(self.max_steering) / self.wheelbase
        print(f"\n[Max Curvature (from steering)]")
        print(f"  Vehicle capability:  {max_curvature:.4f} (1/m) -> min radius: {1/max_curvature:.1f} m")

        # Compare with Unicycle model parameters
        if self.model is not None:
            action_space = self.model.action_space
            # bounds are tuples/lists (or OmegaConf ListConfig), not Tensors
            curv_bounds = np.array(action_space.curvature_bounds)
            # std/mean are Tensors
            curv_std = action_space.curvature_std.cpu().float().item()
            curv_mean = action_space.curvature_mean.cpu().float().item()

            print(f"\n[Unicycle Model Curvature Parameters]")
            print(f"  Bounds:              [{curv_bounds[0]:.4f}, {curv_bounds[1]:.4f}] (1/m)")
            print(f"  Min radius:          {1/max(abs(curv_bounds[0]), abs(curv_bounds[1])):.1f} m")
            print(f"  Std (normalization): {curv_std:.4f}")
            print(f"  Mean:                {curv_mean:.4f}")
            print(f"\n  Note: Small std={curv_std:.4f} means normalized outputs need scaling!")

            # Check acceleration parameters too
            accel_bounds = np.array(action_space.accel_bounds)
            accel_std = action_space.accel_std.cpu().float().item()
            accel_mean = action_space.accel_mean.cpu().float().item()

            print(f"\n[Unicycle Model Acceleration Parameters]")
            print(f"  Bounds:              [{accel_bounds[0]:.2f}, {accel_bounds[1]:.2f}] (m/s²)")
            print(f"  Std (normalization): {accel_std:.2f}")
            print(f"  Mean:                {accel_mean:.2f}")

        # Center of mass
        com = physics.center_of_mass
        print(f"\n[Center of Mass]")
        print(f"  Position:            x={com.x/100:.3f}m, y={com.y/100:.3f}m, z={com.z/100:.3f}m")

        # Vehicle mass
        print(f"\n[Mass]")
        print(f"  Total mass:          {physics.mass:.1f} kg")

        # Bounding box (to understand vehicle origin)
        bbox = self.ego_vehicle.bounding_box
        print(f"\n[Bounding Box & Vehicle Origin]")
        print(f"  Extent (half-size):  x={bbox.extent.x:.3f}m, y={bbox.extent.y:.3f}m, z={bbox.extent.z:.3f}m")
        print(f"  BBox center offset:  x={bbox.location.x:.3f}m, y={bbox.location.y:.3f}m, z={bbox.location.z:.3f}m")
        print(f"  Full length:         {bbox.extent.x * 2:.3f}m")
        print(f"  Full width:          {bbox.extent.y * 2:.3f}m")
        print(f"  Full height:         {bbox.extent.z * 2:.3f}m")

        # Rear axle x position in actor-local frame (CARLA X=forward)
        # = rear bumper x + rear_axle_offset
        self._rear_axle_x_local = (
            bbox.location.x - bbox.extent.x + self._config.control.rear_axle_offset
        )
        print(f"\n[Rear Axle (actor-local frame)]")
        print(f"  Rear bumper x:       {bbox.location.x - bbox.extent.x:.3f} m")
        print(f"  Rear axle x:         {self._rear_axle_x_local:.3f} m  "
              f"(+{self._config.control.rear_axle_offset:.2f}m from bumper)")

        print("\n" + "="*80)
        print("CARLA VEHICLE CONTROL INPUT RANGES")
        print("="*80)
        print(f"  control.steer:       [-1.0, 1.0] (normalized)")
        print(f"  control.throttle:    [0.0, 1.0]")
        print(f"  control.brake:       [0.0, 1.0]")
        print(f"\n  Note: control.steer is NORMALIZED [-1.0, 1.0], NOT in radians!")
        print(f"  Actual angle = control.steer * max_steer_angle")
        print("="*80 + "\n")

        # Initialize decoupled controller with vehicle parameters from config
        ctrl = self._config.control
        self.decoupled_controller = DecoupledController(
            wheelbase=self.wheelbase,
            max_steering=self.max_steering,
            lateral_lookahead=ctrl.lateral_lookahead,
            min_lookahead_distance=ctrl.min_lookahead_distance,
            spline_num_points=ctrl.spline_num_points,
            throttle_gain=ctrl.throttle_gain,
            brake_threshold=ctrl.brake_threshold,
            brake_value=ctrl.brake_value,
        )
        print(f"[Decoupled Controller] Initialized with:")
        print(f"  Lateral lookahead:   {self.decoupled_controller.lateral_lookahead:.1f} m (fixed)")
        print(f"  Min lookahead:       {self.decoupled_controller.min_lookahead_distance:.1f} m")
        print(f"  Spline points:       {self.decoupled_controller.spline_num_points}")

    def _init_video_writer(self) -> None:
        """Initialize video writer for trajectory visualization."""
        # Close existing writer if any
        if self.video_writer is not None:
            try:
                self.video_writer.release()
            except Exception:
                pass

        # Generate filename with segment index
        video_filename = f"trajectory_{self.video_segment_index:03d}.mp4"
        video_path = f"{self.log_dir}/{video_filename}"

        # Video parameters (resolution and FPS from config)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            video_path,
            fourcc,
            self._config.video.fps,
            (self._config.camera.width, self._config.camera.height),
        )
        print(f"Initialized video writer: {video_path}")
        self.video_frame_count = 0

    def _init_camera_params(self) -> None:
        """Initialize camera parameters for projection."""
        # Get front wide camera
        front_camera = self.cameras.get("camera_front_wide_120fov")
        if front_camera is None:
            print("Warning: camera_front_wide_120fov not found")
            return

        # Camera intrinsics from config
        cam = self._config.camera
        self.image_width = cam.width
        self.image_height = cam.height
        self.fov = cam.front_wide_fov

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
        # Front wide camera is mounted at (front_longitudinal_offset, 0, height_offset)
        # with rotation pitch=0, yaw=0, roll=0 (aligned with vehicle)
        self.camera_offset = np.array([
            cam.front_longitudinal_offset, 0.0, cam.height_offset
        ])

        # Camera rotation relative to ego vehicle (identity for front camera)
        self.camera_rotation = spt.Rotation.from_euler('zyx', [0, 0, 0])

        print(f"Camera offset (ego vehicle frame): {self.camera_offset}")
        print(f"Camera rotation: {self.camera_rotation.as_euler('zyx', degrees=True)}")

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

        # Transform from PhysicalAI-AV frame to camera frame
        # PhysicalAI-AV frame (trajectory): X=forward, Y=left, Z=up
        # CARLA frame: X=forward, Y=right, Z=up
        # Need to flip Y axis: Y_physicalai = -Y_carla

        # Step 1: Convert PhysicalAI-AV frame to CARLA frame
        traj_carla = trajectory_local.copy()
        traj_carla[:, 1] = -traj_carla[:, 1]  # Flip Y axis (left -> right)

        # Step 2: Translate to camera position (camera is offset from ego center)
        # Camera offset is in CARLA frame
        traj_cam_carla = traj_carla - self.camera_offset

        # Step 3: Apply camera rotation (identity for front camera)
        traj_cam = self.camera_rotation.inv().apply(traj_cam_carla)

        # Step 4: Filter points behind camera (X > 0 in CARLA camera frame)
        valid_mask = traj_cam[:, 0] > 0.1  # At least 10cm in front

        # Step 5: Convert CARLA camera frame to standard CV camera frame
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

        # Draw reference markers to visualize coordinate system
        # Camera is at [2.0, 0.0, 1.5] from vehicle origin, so origin is likely out of view

        # 1. Draw rear axle position (Unicycle model origin)
        # Estimate rear axle from bounding box (if available)
        try:
            bbox = self.ego_vehicle.bounding_box
            # Rear axle is approximately at rear bumper + small offset
            rear_axle_x = bbox.location.x - bbox.extent.x + 0.5  # ~0.5m from rear bumper
            rear_axle_local = np.array([[rear_axle_x, 0.0, 0.0]])
            rear_axle_2d, rear_axle_valid = self._project_trajectory_to_image(rear_axle_local)
            if rear_axle_valid[0]:
                u_ra, v_ra = int(rear_axle_2d[0, 0]), int(rear_axle_2d[0, 1])
                if 0 <= u_ra < self.image_width and 0 <= v_ra < self.image_height:
                    cv2.drawMarker(img_bgr, (u_ra, v_ra), (255, 0, 255),  # Magenta
                                  cv2.MARKER_CROSS, 25, 3)
                    cv2.putText(img_bgr, "REAR AXLE", (u_ra + 15, v_ra - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        except (AttributeError, RuntimeError):
            pass

        # 2. Draw vehicle front reference (visible in camera view)
        front_ref_local = np.array([[3.0, 0.0, 0.0]])  # 3m forward from origin
        front_ref_2d, front_ref_valid = self._project_trajectory_to_image(front_ref_local)
        if front_ref_valid[0]:
            u_fr, v_fr = int(front_ref_2d[0, 0]), int(front_ref_2d[0, 1])
            if 0 <= u_fr < self.image_width and 0 <= v_fr < self.image_height:
                cv2.drawMarker(img_bgr, (u_fr, v_fr), (0, 255, 255),  # Yellow
                              cv2.MARKER_DIAMOND, 20, 2)
                cv2.putText(img_bgr, "3m FWD", (u_fr + 15, v_fr + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        if points_2d.shape[0] == 0:
            # Add HUD even if no trajectory
            img_with_hud = self._add_hud_overlay(img_bgr)
            self._draw_language_traces(img_with_hud)
            return cv2.cvtColor(img_with_hud, cv2.COLOR_BGR2RGB)

        # Draw trajectory as connected line segments with gradient color
        prev_point = None
        for i, (point, valid) in enumerate(zip(points_2d, valid_mask)):
            if not valid:
                prev_point = None
                continue

            u, v = int(point[0]), int(point[1])

            # Check if point is within image bounds
            if 0 <= u < self.image_width and 0 <= v < self.image_height:
                # Color gradient from green (near) to red (far)
                ratio = i / len(points_2d)
                color = (0, int(255 * (1 - ratio)), int(255 * ratio))  # BGR: Green -> Yellow -> Red

                # Draw point with thicker circle for better visibility
                # Emphasize trajectory start point (index 0)
                if i == 0:
                    cv2.circle(img_bgr, (u, v), 15, (255, 255, 0), 3)  # Large cyan circle
                    cv2.circle(img_bgr, (u, v), 8, color, -1)  # Filled center
                    cv2.putText(img_bgr, "TRAJ START", (u + 20, v),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                else:
                    cv2.circle(img_bgr, (u, v), 6, color, -1)

                # Draw index number for first, middle, and last few points
                if i < 3 or i > len(points_2d) - 4 or i % 10 == 0:
                    cv2.putText(img_bgr, str(i), (u + 8, v - 8),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

                # Draw line from previous point
                if prev_point is not None:
                    prev_u, prev_v = int(prev_point[0]), int(prev_point[1])
                    if 0 <= prev_u < self.image_width and 0 <= prev_v < self.image_height:
                        cv2.line(img_bgr, (prev_u, prev_v), (u, v), color, 3)

                prev_point = point
            else:
                prev_point = None

        # Add HUD overlay with vehicle info
        img_with_hud = self._add_hud_overlay(img_bgr)

        # Add language traces overlay
        self._draw_language_traces(img_with_hud)

        # Convert back to RGB
        img_rgb = cv2.cvtColor(img_with_hud, cv2.COLOR_BGR2RGB)
        return img_rgb

    def _draw_language_traces(self, image_bgr: np.ndarray) -> None:
        """Draw language traces (CoT, meta_action, answer) on HUD.

        Args:
            image_bgr: Image in BGR format
        """
        # Text box parameters (left side of screen)
        box_x = 20
        box_y = 200  # Below existing HUD
        box_width = 600
        box_height = 400

        # Create semi-transparent background
        overlay = image_bgr.copy()
        cv2.rectangle(overlay, (box_x, box_y),
                     (box_x + box_width, box_y + box_height),
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, image_bgr, 0.4, 0, image_bgr)

        # Draw border
        cv2.rectangle(image_bgr, (box_x, box_y),
                     (box_x + box_width, box_y + box_height),
                     (100, 200, 255), 2)

        # Font settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        line_height = 20

        # Helper function to wrap text
        def wrap_text(text: str, max_width: int = 70) -> list[str]:
            """Wrap text to fit within max_width characters."""
            words = text.split()
            lines = []
            current_line = ""
            for word in words:
                test_line = current_line + " " + word if current_line else word
                if len(test_line) <= max_width:
                    current_line = test_line
                else:
                    if current_line:
                        lines.append(current_line)
                    current_line = word
            if current_line:
                lines.append(current_line)
            return lines

        # Display CoT
        y_offset = box_y + 25
        cv2.putText(image_bgr, "Chain-of-Thought:", (box_x + 10, y_offset),
                   font, font_scale, (0, 255, 255), font_thickness)
        y_offset += line_height

        if self.latest_cot:
            cot_lines = wrap_text(self.latest_cot, max_width=70)
            for i, line in enumerate(cot_lines[:8]):  # Max 8 lines for CoT
                cv2.putText(image_bgr, line, (box_x + 15, y_offset),
                           font, font_scale, (255, 255, 255), font_thickness)
                y_offset += line_height
            if len(cot_lines) > 8:
                cv2.putText(image_bgr, "...", (box_x + 15, y_offset),
                           font, font_scale, (150, 150, 150), font_thickness)
                y_offset += line_height
        else:
            cv2.putText(image_bgr, "(No CoT available)", (box_x + 15, y_offset),
                       font, font_scale, (150, 150, 150), font_thickness)
            y_offset += line_height

        # Add spacing
        y_offset += 10

        # Display Meta Action
        cv2.putText(image_bgr, "Meta Action:", (box_x + 10, y_offset),
                   font, font_scale, (0, 255, 255), font_thickness)
        y_offset += line_height

        if self.latest_meta_action:
            meta_lines = wrap_text(self.latest_meta_action, max_width=70)
            for line in meta_lines[:3]:  # Max 3 lines for meta action
                cv2.putText(image_bgr, line, (box_x + 15, y_offset),
                           font, font_scale, (255, 255, 255), font_thickness)
                y_offset += line_height
        else:
            cv2.putText(image_bgr, "(None)", (box_x + 15, y_offset),
                       font, font_scale, (150, 150, 150), font_thickness)
            y_offset += line_height

        # Add spacing
        y_offset += 10

        # Display Answer
        cv2.putText(image_bgr, "Answer:", (box_x + 10, y_offset),
                   font, font_scale, (0, 255, 255), font_thickness)
        y_offset += line_height

        if self.latest_answer:
            answer_lines = wrap_text(self.latest_answer, max_width=70)
            for line in answer_lines[:3]:  # Max 3 lines for answer
                cv2.putText(image_bgr, line, (box_x + 15, y_offset),
                           font, font_scale, (255, 255, 255), font_thickness)
                y_offset += line_height
        else:
            cv2.putText(image_bgr, "(None)", (box_x + 15, y_offset),
                       font, font_scale, (150, 150, 150), font_thickness)

    def _add_hud_overlay(self, image_bgr: np.ndarray) -> np.ndarray:
        """Add HUD overlay with vehicle information.

        Args:
            image_bgr: Image in BGR format

        Returns:
            Image with HUD overlay in BGR format
        """
        # Check if vehicle is still alive before accessing
        try:
            if not self.ego_vehicle.is_alive:
                return image_bgr
        except RuntimeError:
            # Actor already destroyed
            return image_bgr

        # Get current vehicle state
        try:
            velocity = self.ego_vehicle.get_velocity()
            speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
            location = self.ego_vehicle.get_transform().location
            control = self.ego_vehicle.get_control()
            steer = control.steer
        except RuntimeError:
            # Actor destroyed during access
            return image_bgr

        # Create semi-transparent overlay for HUD
        overlay = image_bgr.copy()

        # Draw background rectangle for HUD (top-left corner, increased height for 5 lines)
        cv2.rectangle(overlay, (10, 10), (450, 180), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.4, image_bgr, 0.6, 0, image_bgr)

        # Font settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2
        line_height = 30

        # Draw HUD text
        y_offset = 35
        cv2.putText(image_bgr, f"Step: {self.step_count}", (20, y_offset),
                    font, font_scale, (0, 255, 255), font_thickness)

        y_offset += line_height
        cv2.putText(image_bgr, f"Speed: {speed:.1f} m/s ({speed*3.6:.1f} km/h)",
                    (20, y_offset), font, font_scale, (0, 255, 255), font_thickness)

        y_offset += line_height
        cv2.putText(image_bgr, f"Position: ({location.x:.1f}, {location.y:.1f})",
                    (20, y_offset), font, font_scale, (0, 255, 255), font_thickness)

        y_offset += line_height
        speed_limit = self._get_carla_speed_limit()
        cv2.putText(image_bgr, f"Speed Limit: {speed_limit:.1f} m/s ({speed_limit*3.6:.0f} km/h)",
                    (20, y_offset), font, font_scale, (0, 255, 255), font_thickness)

        y_offset += line_height
        cv2.putText(image_bgr, f"Steer: {steer:+.3f}",
                    (20, y_offset), font, font_scale, (0, 255, 255), font_thickness)

        # Add Bird's Eye View (BEV) of trajectory in top-right corner
        if self.predicted_trajectory is not None and len(self.predicted_trajectory) > 0:
            self._draw_bev_trajectory(image_bgr)

        return image_bgr

    def _draw_bev_trajectory(self, image_bgr: np.ndarray) -> None:
        """Draw Bird's Eye View of trajectory in top-right corner.

        Args:
            image_bgr: Image in BGR format
        """
        # BEV parameters
        bev_width = 300
        bev_height = 400
        bev_margin = 20
        bev_x = self.image_width - bev_width - bev_margin
        bev_y = bev_margin

        # Create semi-transparent background
        overlay = image_bgr.copy()
        cv2.rectangle(overlay, (bev_x, bev_y),
                     (bev_x + bev_width, bev_y + bev_height),
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, image_bgr, 0.5, 0, image_bgr)

        # Draw border
        cv2.rectangle(image_bgr, (bev_x, bev_y),
                     (bev_x + bev_width, bev_y + bev_height),
                     (255, 255, 255), 2)

        # Add title
        cv2.putText(image_bgr, "Bird's Eye View", (bev_x + 10, bev_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Scale trajectory to fit BEV (assume max 20m forward, 5m lateral)
        max_forward = 20.0  # meters
        max_lateral = 5.0   # meters
        scale_x = bev_height / max_forward
        scale_y = bev_width / (2 * max_lateral)

        # Center of BEV (ego vehicle position)
        center_x = bev_x + bev_width // 2
        center_y = bev_y + bev_height - 20  # Bottom with margin

        # Draw ego vehicle (triangle pointing up)
        ego_size = 15
        ego_pts = np.array([
            [center_x, center_y - ego_size],      # Front
            [center_x - ego_size//2, center_y + ego_size//2],  # Rear left
            [center_x + ego_size//2, center_y + ego_size//2],  # Rear right
        ], dtype=np.int32)
        cv2.fillPoly(image_bgr, [ego_pts], (0, 255, 255))

        # Draw trajectory waypoints
        for i, waypoint in enumerate(self.predicted_trajectory):
            # Transform to BEV coordinates
            # waypoint: [x_forward, y_left, z_up]
            x_forward = waypoint[0]
            y_left = waypoint[1]

            # BEV coordinates (flip Y to match screen coordinates)
            bev_point_x = int(center_x - y_left * scale_y)
            bev_point_y = int(center_y - x_forward * scale_x)

            # Check if point is within BEV bounds
            if (bev_x < bev_point_x < bev_x + bev_width and
                bev_y + 30 < bev_point_y < bev_y + bev_height):
                # Color gradient from green (near) to red (far)
                ratio = i / len(self.predicted_trajectory)
                color = (0, int(255 * (1 - ratio)), int(255 * ratio))

                # Draw waypoint
                cv2.circle(image_bgr, (bev_point_x, bev_point_y), 3, color, -1)

    def update(self, world_snapshot: Any, camera_images: dict[str, np.ndarray]) -> None:
        """Update controller state and compute control commands.

        Args:
            world_snapshot: CARLA world snapshot
            camera_images: Dictionary mapping camera names to RGB images (H, W, 3)
        """
        self.step_count += 1
        self.world_snapshot = world_snapshot

        # Run model inference at control frequency
        # steps_per_control = sim_hz / control_hz = (1/delta_seconds) / control_frequency
        sim_hz = 1.0 / self._config.simulation.delta_seconds
        steps_per_control = round(sim_hz / self.control_frequency)
        if self.step_count - self.last_control_step >= steps_per_control:
            # Update ego state at 10Hz (same as model training frequency)
            self._update_ego_state()

            # Store images, update rolling buffer, and run inference
            self.current_images = camera_images
            self._update_camera_buffer(camera_images)
            self._run_inference()
            self.last_control_step = self.step_count

        # Apply control EVERY frame (using latest prediction from model)
        # This allows smooth control at simulation rate (20Hz) while inference runs at 10Hz
        self._apply_decoupled_control()

        # Save visualization video
        if self.save_video and self.video_writer is not None:
            self._save_visualization_frame(camera_images)

    def _update_camera_buffer(self, camera_images: dict[str, np.ndarray]) -> None:
        """Update rolling frame buffer with the latest camera images.

        Converts each image to a (C, H, W) tensor and appends it to the
        per-camera deque (maxlen=NUM_FRAME_HISTORY).  The buffer accumulates
        frames over time at the inference frequency (10 Hz), mirroring the
        training data layout: [t0-0.3s, t0-0.2s, t0-0.1s, t0].

        Args:
            camera_images: Dictionary mapping camera names to RGB images (H, W, 3)
        """
        for cam_name, image in camera_images.items():
            if cam_name not in self.camera_frame_buffer:
                self.camera_frame_buffer[cam_name] = deque(maxlen=self.NUM_FRAME_HISTORY)
            img_tensor = torch.from_numpy(image.copy()).float()
            img_tensor = rearrange(img_tensor, "h w c -> c h w")
            self.camera_frame_buffer[cam_name].append(img_tensor)

    def _save_visualization_frame(self, camera_images: dict[str, np.ndarray]) -> None:
        """Save current frame with trajectory visualization to video.

        Args:
            camera_images: Dictionary mapping camera names to RGB images
        """
        # Check if we need to start a new video segment
        if self.video_frame_count >= self.frames_per_segment:
            self.video_segment_index += 1
            self._init_video_writer()

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
        self.video_frame_count += 1

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

    def _analyze_trajectory_offset(self) -> None:
        """Analyze coordinate offset between trajectory start and expected origins.

        This helps diagnose if the trajectory is offset from the expected vehicle origin,
        which can cause 'curves too late' behavior.
        """
        if self.predicted_trajectory is None or len(self.predicted_trajectory) == 0:
            return

        # Only print every 20 steps to avoid spam
        if self.step_count % 20 != 0:
            return

        try:
            # 1. Vehicle origin (by definition)
            vehicle_origin = np.array([0.0, 0.0, 0.0])

            # 2. Estimated rear axle position (Unicycle model origin)
            bbox = self.ego_vehicle.bounding_box
            rear_axle_x = bbox.location.x - bbox.extent.x + self._config.control.rear_axle_offset
            rear_axle_pos = np.array([rear_axle_x, 0.0, 0.0])

            # 3. Trajectory start point
            traj_start = self.predicted_trajectory[0]  # (3,) [x, y, z]

            # 4. Calculate offsets
            offset_from_vehicle = traj_start - vehicle_origin
            offset_from_rear_axle = traj_start - rear_axle_pos

            print("\n" + "="*70)
            print("TRAJECTORY COORDINATE OFFSET ANALYSIS")
            print("="*70)
            print(f"Vehicle origin (CARLA):     ({vehicle_origin[0]:6.3f}, {vehicle_origin[1]:6.3f}, {vehicle_origin[2]:6.3f}) m")
            print(f"Rear axle (estimated):      ({rear_axle_pos[0]:6.3f}, {rear_axle_pos[1]:6.3f}, {rear_axle_pos[2]:6.3f}) m")
            print(f"Trajectory start [0]:       ({traj_start[0]:6.3f}, {traj_start[1]:6.3f}, {traj_start[2]:6.3f}) m")
            print(f"\nOffset from vehicle origin: ({offset_from_vehicle[0]:6.3f}, {offset_from_vehicle[1]:6.3f}, {offset_from_vehicle[2]:6.3f}) m")
            print(f"Offset from rear axle:      ({offset_from_rear_axle[0]:6.3f}, {offset_from_rear_axle[1]:6.3f}, {offset_from_rear_axle[2]:6.3f}) m")

            # Diagnose issues
            forward_offset_from_rear = offset_from_rear_axle[0]
            if abs(forward_offset_from_rear) > 0.5:
                print(f"\n⚠ WARNING: Trajectory starts {forward_offset_from_rear:.2f}m from rear axle!")
                print(f"⚠ Expected offset: ~0.0m (Unicycle model assumes rear axle origin)")
                if forward_offset_from_rear > 0.5:
                    print(f"⚠ Forward offset explains 'curves too late' behavior.")
                    print(f"⚠ Consider applying correction: trajectory -= [{forward_offset_from_rear:.3f}, 0, 0]")
                else:
                    print(f"⚠ Backward offset may cause 'curves too early' behavior.")
            else:
                print(f"\n✓ Trajectory origin alignment looks good (offset: {forward_offset_from_rear:.3f}m)")

            print("="*70 + "\n")

        except (AttributeError, RuntimeError, IndexError):
            # This is diagnostic only - safe to ignore errors
            pass

    def _run_inference(self) -> None:
        """Run Alpamayo R1 model inference on current observations."""
        if self.model is None or self.processor is None or len(self.current_images) == 0:
            return

        try:
            # Prepare input data
            model_input = self._prepare_model_input()

            # Run inference with inference_mode (more efficient than no_grad) and autocast
            # inference_mode disables view tracking and version counter, reducing memory overhead
            inf = self._config.inference
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                pred_xyz, pred_rot, extra = self.model.sample_trajectories_from_data_with_vlm_rollout(
                    data=model_input,
                    top_p=inf.top_p,
                    temperature=inf.temperature,
                    num_traj_samples=inf.num_traj_samples,
                    max_generation_length=inf.max_generation_length,
                    return_extra=True,
                )

            # Get sampled action from extra (already a Tensor on CPU)
            sampled_action = extra["sampled_action"]  # (1, 1, 1, 64, 2) Tensor
            sampled_action_tensor = sampled_action[0, 0, 0].float().to("cuda")  # (64, 2)

            # Extract language traces (CoT, meta_action, answer)
            # Shape: [B, num_traj_sets, num_traj_samples] = [1, 1, 1]
            self.latest_cot = extra["cot"][0, 0, 0] if "cot" in extra else ""
            self.latest_meta_action = extra["meta_action"][0, 0, 0] if "meta_action" in extra else ""
            self.latest_answer = extra["answer"][0, 0, 0] if "answer" in extra else ""

            # Debug log (every 10 steps)
            if self.step_count % 10 == 0:
                print(f"\n[CoT] {self.latest_cot[:100]}...")  # First 100 chars
                if self.latest_meta_action:
                    print(f"[Meta Action] {self.latest_meta_action}")

            # Debug: Print action statistics
            accel_normalized = sampled_action_tensor[:, 0].cpu().numpy()
            curvature_normalized = sampled_action_tensor[:, 1].cpu().numpy()

            # Denormalize to get real values (convert from BFloat16 to float32 first)
            accel_std = self.model.action_space.accel_std.cpu().float().numpy()
            accel_mean = self.model.action_space.accel_mean.cpu().float().numpy()
            curv_std = self.model.action_space.curvature_std.cpu().float().numpy()
            curv_mean = self.model.action_space.curvature_mean.cpu().float().numpy()

            accel_real = accel_normalized * accel_std + accel_mean
            curvature_real = curvature_normalized * curv_std + curv_mean

            print(f"[Action] Accel: [{accel_real.min():.3f}, {accel_real.max():.3f}] m/s²")
            print(f"[Action] Curvature: [{curvature_real.min():.4f}, {curvature_real.max():.4f}] 1/m "
                  f"-> radius: {1/max(abs(curvature_real.min()), abs(curvature_real.max()), 1e-6):.1f}m")

            # Get current actual speed
            current_velocity = self.ego_vehicle.get_velocity()
            current_speed = np.sqrt(
                current_velocity.x**2 + current_velocity.y**2 + current_velocity.z**2
            )

            # Prepare correct initial state with actual speed
            t0_states = {"v": torch.tensor([[current_speed]], device="cuda", dtype=torch.float32)}

            # Get ego history for action_to_traj
            ego_history_xyz = model_input["ego_history_xyz"]  # (1, 1, 16, 3)
            ego_history_rot = model_input["ego_history_rot"]  # (1, 1, 16, 3, 3)

            # Recompute trajectory with correct initial speed
            pred_xyz_corrected, pred_rot_corrected = self.model.action_space.action_to_traj(
                sampled_action_tensor.unsqueeze(0),  # (1, 64, 2)
                ego_history_xyz[:, -1],  # (1, 16, 3)
                ego_history_rot[:, -1],  # (1, 16, 3, 3)
                t0_states=t0_states
            )

            # Extract corrected trajectory
            # Trajectory is already in rear-axle local frame because ego_history_xyz
            # was constructed with rear axle as origin in _prepare_model_input().
            self.predicted_trajectory = pred_xyz_corrected.cpu().numpy()[0]  # (num_timesteps, 3)

            # Delete intermediate tensors to free memory immediately
            del pred_xyz, pred_rot, extra, model_input, sampled_action_tensor
            del pred_xyz_corrected, pred_rot_corrected

            # Clear CUDA cache to free memory for next inference
            torch.cuda.empty_cache()

            # Log GPU memory usage every 10 inferences
            if self.step_count % 20 == 0:
                allocated = torch.cuda.memory_allocated(0) / 1e9
                reserved = torch.cuda.memory_reserved(0) / 1e9
                print(f"[GPU Memory] Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")

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

        # Build (N_cameras, NUM_FRAME_HISTORY, C, H, W) from rolling buffer,
        # then flatten to (N_cameras * NUM_FRAME_HISTORY, C, H, W) to match
        # the training data layout used in test_inference.py.
        camera_frames = []
        for cam_name, _ in sorted_cameras:
            buf = self.camera_frame_buffer.get(cam_name)
            if buf is None or len(buf) == 0:
                # Buffer not yet filled: repeat the current image
                img = torch.from_numpy(self.current_images[cam_name].copy()).float()
                img = rearrange(img, "h w c -> c h w")
                frames_for_cam = [img] * self.NUM_FRAME_HISTORY
            else:
                frames_for_cam = list(buf)
                # Pad front with oldest frame when buffer has fewer than NUM_FRAME_HISTORY entries
                while len(frames_for_cam) < self.NUM_FRAME_HISTORY:
                    frames_for_cam.insert(0, frames_for_cam[0])
            camera_frames.append(torch.stack(frames_for_cam, dim=0))  # (NUM_FRAME_HISTORY, C, H, W)

        # (N_cameras, NUM_FRAME_HISTORY, C, H, W) -> (N_cameras * NUM_FRAME_HISTORY, C, H, W)
        images = torch.stack(camera_frames, dim=0).flatten(0, 1)

        # Get current vehicle speed
        current_velocity = self.ego_vehicle.get_velocity()
        current_speed = np.sqrt(
            current_velocity.x**2 + current_velocity.y**2 + current_velocity.z**2
        )

        # Get target speed (speed limit with reduction factor applied)
        target_speed = self._get_carla_speed_limit() * self._get_speed_reduction_factor()

        # Create messages from images using helper
        # helper.create_message expects (N, C, H, W)
        messages = helper.create_message(images, current_speed=current_speed, target_speed=target_speed)

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

            # Transform to local frame with rear axle as origin
            # (Unicycle model expects rear axle as coordinate origin)
            current_rot = spt.Rotation.from_matrix(history_rot[-1])
            current_rot_inv = current_rot.inv()

            # Rear axle world position at the current timestep
            rear_axle_world = history_xyz[-1] + current_rot.apply(
                np.array([self._rear_axle_x_local, 0.0, 0.0])
            )

            history_xyz_local = current_rot_inv.apply(history_xyz - rear_axle_world)

            # Convert CARLA coordinate system (X=forward, Y=right) to
            # model's expected coordinate system (X=forward, Y=left)
            history_xyz_local[:, 1] = -history_xyz_local[:, 1]

            history_rot_local = (
                current_rot_inv * spt.Rotation.from_matrix(history_rot)
            ).as_matrix()

            # Convert rotation matrices from CARLA (Y=right) to PhysicalAI-AV (Y=left) convention.
            # Positions were already converted above (history_xyz_local[:, 1] *= -1).
            # For rotation matrices, the equivalent change of basis is:
            #   R_physicalai = T @ R_carla @ T,  where T = diag(1, -1, 1)
            # which negates the Y row then the Y column (double-negation leaves [1,1] unchanged).
            history_rot_local[:, 1, :] *= -1  # negate Y row
            history_rot_local[:, :, 1] *= -1  # negate Y column

            # Debug: Log ego history to analyze trajectory curvature (minimal version to save memory)
            if self.step_count % 50 == 0:  # Reduced frequency: every 50 steps
                print(f"\n[DEBUG Ego History] Step: {self.step_count}, Length: {num_history}")
                # Only log last position for brevity
                if num_history > 0:
                    print(f"  Last local pos (Alpamayo Y=left): [{history_xyz_local[-1, 0]:7.2f}, {history_xyz_local[-1, 1]:7.2f}, {history_xyz_local[-1, 2]:7.2f}]")
                # Simplified curvature calculation (no intermediate arrays stored)
                if num_history >= 3:
                    dx = np.gradient(history_xyz_local[:, 0])
                    dy = np.gradient(history_xyz_local[:, 1])
                    curvatures = np.abs(dx * np.gradient(dy) - dy * np.gradient(dx)) / np.power(dx**2 + dy**2 + 1e-6, 1.5)
                    print(f"  Ego history curvature: max={np.max(curvatures):.4f}")
                print(f"  Speed: {current_speed:.2f}/{target_speed:.2f} m/s")

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

    def _calculate_trajectory_length(self, trajectory: np.ndarray) -> float:
        """Calculate total length of trajectory polyline in XY plane.

        Args:
            trajectory: (N, 3) array [x, y, z]

        Returns:
            total_length: Total polyline length in meters
        """
        positions_2d = trajectory[:, :2]
        deltas = np.diff(positions_2d, axis=0)
        distances = np.linalg.norm(deltas, axis=1)
        total_length = np.sum(distances)
        return total_length

    def _get_speed_reduction_factor(self) -> float:
        """Calculate speed reduction factor based on trajectory length.

        If predicted trajectory is shorter than expected (indicating
        stopping or slowing intention), reduce speed proportionally.

        Returns:
            speed_factor: Multiplier for speed limit (0.1 to 1.0)
        """
        if self.predicted_trajectory is None or len(self.predicted_trajectory) < 20:
            return 1.0

        # Use 2 seconds ahead (20 waypoints at 0.1s interval)
        trajectory_subset = self.predicted_trajectory[:20]
        actual_length = self._calculate_trajectory_length(trajectory_subset)

        # Expected length based on current road speed limit
        reference_speed = self._get_carla_speed_limit()  # Use actual speed limit
        time_horizon = 2.0  # seconds
        expected_length = reference_speed * time_horizon

        # Calculate length ratio
        length_ratio = actual_length / expected_length

        # Apply speed reduction if trajectory is significantly shorter
        ctrl = self._config.control
        reduction_threshold = ctrl.speed_reduction_threshold
        if length_ratio < reduction_threshold:
            # Linear reduction from threshold to minimum
            speed_factor = max(ctrl.min_speed_factor, length_ratio / reduction_threshold)

            if self.step_count % 20 == 0:
                print(f"[Speed Reduction] Trajectory length: {actual_length:.1f}m / {expected_length:.1f}m "
                      f"(ratio: {length_ratio:.2f}) -> factor: {speed_factor:.2f}")

            return speed_factor
        else:
            return 1.0

    def _calculate_curvature_speed_limit(self, max_lateral_accel: float = 4.0) -> float:
        """Calculate safe speed limit based on trajectory curvature.

        Uses the relationship: lateral_accel = v² × curvature
        Therefore: v_safe = sqrt(max_lateral_accel / curvature)

        Args:
            max_lateral_accel: Maximum comfortable lateral acceleration [m/s²]
                              Typical values: 2-4 m/s² (comfort), 6-8 m/s² (sport)

        Returns:
            speed_limit: Safe speed for the trajectory's curvature [m/s]
                        Returns a large value (100 m/s) if trajectory is nearly straight
        """
        if self.predicted_trajectory is None or len(self.predicted_trajectory) < 3:
            return 100.0  # No limit if no trajectory

        # Analyze curvature in lookahead range (next 3 seconds, 30 waypoints)
        lookahead_steps = min(30, len(self.predicted_trajectory))
        trajectory_subset = self.predicted_trajectory[:lookahead_steps]

        # Calculate curvature using finite differences
        # positions: (x, y) in 2D
        positions = trajectory_subset[:, :2]  # (N, 2)

        # First derivatives (velocity direction)
        dx = np.gradient(positions[:, 0])
        dy = np.gradient(positions[:, 1])

        # Second derivatives (acceleration direction)
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)

        # Curvature formula: κ = |dx*ddy - dy*ddx| / (dx² + dy²)^(3/2)
        numerator = np.abs(dx * ddy - dy * ddx)
        denominator = np.power(dx**2 + dy**2, 1.5)

        # Avoid division by zero
        denominator = np.maximum(denominator, 1e-6)
        curvatures = numerator / denominator

        # Get maximum curvature in lookahead range
        max_curvature = np.max(curvatures)

        # Calculate safe speed
        # If curvature is very small (nearly straight), no speed limit
        min_curvature_threshold = 0.01  # 1/m (radius > 100m)
        if max_curvature < min_curvature_threshold:
            return 100.0  # No limit for straight roads

        # v_safe = sqrt(lateral_accel / curvature)
        safe_speed = np.sqrt(max_lateral_accel / max_curvature)

        # Log speed limit calculation (every 20 steps)
        if self.step_count % 20 == 0:
            radius = 1.0 / max_curvature if max_curvature > 1e-6 else float('inf')
            print(f"[Curvature Speed Limit] Max curvature: {max_curvature:.4f} (1/m), "
                  f"Radius: {radius:.1f}m → Safe speed: {safe_speed:.1f} m/s ({safe_speed*3.6:.1f} km/h)")

        return safe_speed

    def _get_carla_speed_limit(self) -> float:
        """Get speed limit at current ego vehicle location from CARLA API.

        Returns:
            speed_limit: Speed limit in m/s (80% of OpenDRIVE limit)
        """
        speed_limit_kmh = self.ego_vehicle.get_speed_limit()
        speed_limit_ms = speed_limit_kmh / 3.6

        # Apply speed_limit_factor to OpenDRIVE speed limit for safer driving
        speed_limit_ms *= self._config.control.speed_limit_factor

        speed_limit_ms = min(speed_limit_ms, self.max_speed)

        if self.step_count % 20 == 0:
            print(f"[Speed Limit] {speed_limit_kmh:.0f} km/h → {speed_limit_ms:.1f} m/s (80%)")

        return speed_limit_ms

    def _apply_control(self) -> None:
        """Apply control commands based on predicted trajectory."""
        if self.predicted_trajectory is None or len(self.predicted_trajectory) == 0:
            # Fallback: maintain current speed
            self._apply_simple_control()
            return

        # Get speed limit from CARLA
        target_speed = self._get_carla_speed_limit()

        # Apply speed reduction based on trajectory length
        speed_reduction_factor = self._get_speed_reduction_factor()
        target_speed *= speed_reduction_factor

        # Apply curvature-based speed limit (for safe cornering)
        curvature_speed_limit = self._calculate_curvature_speed_limit(
            max_lateral_accel=self._config.control.max_lateral_accel
        )
        target_speed = min(target_speed, curvature_speed_limit)

        # Lateral control using Pure Pursuit algorithm
        # Get current speed for speed-dependent lookahead
        current_velocity = self.ego_vehicle.get_velocity()
        current_speed = np.sqrt(
            current_velocity.x**2 + current_velocity.y**2 + current_velocity.z**2
        )

        # Calculate lookahead distance (speed-dependent)
        lookahead_distance_desired = max(
            current_speed * self.lookahead_time,
            self.min_lookahead_distance
        )

        # Limit lookahead to trajectory range (90% of max to avoid overshoot)
        # This is important when trajectory is short (e.g., stopping intention)
        max_traj_distance = np.sqrt(
            self.predicted_trajectory[-1, 0]**2 + self.predicted_trajectory[-1, 1]**2
        )
        lookahead_distance_desired = min(lookahead_distance_desired, max_traj_distance * 0.9)

        # Find waypoint closest to desired lookahead distance
        best_idx = 0
        min_diff = float('inf')
        for i in range(len(self.predicted_trajectory)):
            wp = self.predicted_trajectory[i]
            wp_distance = np.sqrt(wp[0]**2 + wp[1]**2)
            diff = abs(wp_distance - lookahead_distance_desired)
            if diff < min_diff:
                min_diff = diff
                best_idx = i

        target_point = self.predicted_trajectory[best_idx]
        target_x = target_point[0]
        target_y = target_point[1]

        # Calculate actual lookahead distance
        lookahead_distance = np.sqrt(target_x**2 + target_y**2)

        # Pure pursuit: calculate curvature
        # curvature = 2 * sin(alpha) / L, where sin(alpha) ≈ lateral_error / L
        # Simplified: curvature = 2 * lateral_error / L^2
        # Model coords: Y=left (positive = target on left)
        # CARLA control: steer positive = turn left, negative = turn right
        # Therefore: positive target_y (left) → positive curvature → positive steer (turn left)
        steering_angle_rad = 0.0  # Initialize for debug output
        if lookahead_distance > 0.1:  # Avoid division by zero
            curvature = 2.0 * target_y / (lookahead_distance**2)

            # Convert curvature to steering angle (in radians)
            # steering_angle = atan(wheelbase * curvature)
            steering_angle_rad = np.arctan(self.wheelbase * curvature)

            # Clamp to max steering angle (radians)
            steering_angle_rad = np.clip(steering_angle_rad, -self.max_steering, self.max_steering)

            # Normalize to CARLA control range [-1.0, 1.0] with sign flip
            # CARLA steer+ = right, model Y=left so positive curvature = left → negate
            steering = -steering_angle_rad / self.max_steering
        else:
            # Too close, go straight
            steering = 0.0

        # Longitudinal control (speed)
        ctrl = self._config.control
        speed_error = target_speed - current_speed
        throttle = np.clip(ctrl.throttle_gain * speed_error, 0.0, 1.0)
        brake = 0.0 if speed_error > ctrl.brake_threshold else ctrl.brake_value

        # Apply control
        control = carla.VehicleControl()
        control.hand_brake = False  # Ensure handbrake is off
        control.throttle = float(throttle)
        control.steer = float(steering)
        control.brake = float(brake)

        # Check current gear and set to forward if needed
        current_control = self.ego_vehicle.get_control()
        if current_control.gear == 0:
            # If in neutral, manually set gear to 1 (forward)
            control.manual_gear_shift = True
            control.gear = 1
        else:
            # Otherwise use automatic transmission
            control.manual_gear_shift = False

        # Get vehicle state for debugging
        vehicle_transform = self.ego_vehicle.get_transform()
        vehicle_location = vehicle_transform.location
        vehicle_control_state = self.ego_vehicle.get_control()

        # Get camera state for debugging
        front_camera = self.cameras.get("camera_front_wide_120fov")
        if front_camera is not None:
            try:
                camera_transform = front_camera.get_transform()
                camera_location = camera_transform.location
                # Note: Sensors don't have is_attached_to() method
                camera_attached = True  # Assume attached if we can get transform
            except RuntimeError:
                camera_location = None
                camera_attached = False
        else:
            camera_location = None
            camera_attached = False

        # Debug output
        steering_deg = np.rad2deg(steering_angle_rad)
        print(f"[Control] Step: {self.step_count:4d} | "
              f"WP[{best_idx:2d}]: ({target_x:5.2f}, {target_y:5.2f}) | "
              f"Lookahead: {lookahead_distance:.1f}m | "
              f"Speed: {current_speed:4.1f}/{target_speed:4.1f} m/s | "
              f"Steer: {steering_deg:+5.1f}° ({control.steer:+.3f}) | "
              f"Throttle: {control.throttle:.3f}")

        self.ego_vehicle.apply_control(control)

    def _apply_decoupled_control(self) -> None:
        """Apply decoupled lateral and longitudinal control.

        Uses DecoupledController class for:
        - Lateral control: Fixed lookahead distance on spline-interpolated trajectory
        - Longitudinal control: Speed control based on curvature and trajectory length
        """
        if self.predicted_trajectory is None or len(self.predicted_trajectory) == 0:
            # Fallback: maintain current speed
            self._apply_simple_control()
            return

        if self.decoupled_controller is None:
            print("[Warning] DecoupledController not initialized, using simple control")
            self._apply_simple_control()
            return

        # --- Step 1: Compute target speed (Longitudinal) ---
        # Get speed limit from CARLA
        target_speed = self._get_carla_speed_limit()

        # Apply speed reduction based on trajectory length
        speed_reduction_factor = self._get_speed_reduction_factor()
        target_speed *= speed_reduction_factor

        # Apply curvature-based speed limit (for safe cornering)
        curvature_speed_limit = self._calculate_curvature_speed_limit(
            max_lateral_accel=self._config.control.max_lateral_accel
        )
        target_speed = min(target_speed, curvature_speed_limit)

        # Get current speed
        current_velocity = self.ego_vehicle.get_velocity()
        current_speed = np.sqrt(
            current_velocity.x**2 + current_velocity.y**2 + current_velocity.z**2
        )

        # --- Step 2: Compute control commands using DecoupledController ---
        control_output = self.decoupled_controller.compute_control(
            trajectory=self.predicted_trajectory,
            target_speed=target_speed,
            current_speed=current_speed,
        )

        # Extract control values
        steering = control_output["steering"]
        throttle = control_output["throttle"]
        brake = control_output["brake"]
        steering_angle_rad = control_output["steering_angle_rad"]
        target_point = control_output["target_point"]
        target_idx = control_output["target_idx"]
        lookahead_distance = control_output["lookahead_distance"]

        # --- Step 3: Apply Control ---
        control = carla.VehicleControl()
        control.hand_brake = False
        control.throttle = float(throttle)
        control.steer = float(steering)
        control.brake = float(brake)

        # Check current gear and set to forward if needed
        current_control = self.ego_vehicle.get_control()
        if current_control.gear == 0:
            control.manual_gear_shift = True
            control.gear = 1
        else:
            control.manual_gear_shift = False

        # --- Step 4: Debug Output ---
        steering_deg = np.rad2deg(steering_angle_rad)
        target_x, target_y = target_point[0], target_point[1]
        print(f"[Decoupled Control] Step: {self.step_count:4d} | "
              f"Lateral: WP[{target_idx:3d}] ({target_x:5.2f}, {target_y:5.2f}), "
              f"Lookahead: {lookahead_distance:.1f}m | "
              f"Longitudinal: Speed {current_speed:4.1f}/{target_speed:4.1f} m/s | "
              f"Steer: {steering_deg:+5.1f}° ({control.steer:+.3f}) | "
              f"Throttle: {control.throttle:.3f}")

        self.ego_vehicle.apply_control(control)

    def _apply_simple_control(self) -> None:
        """Apply simple speed control when no trajectory available."""
        current_velocity = self.ego_vehicle.get_velocity()
        current_speed = np.sqrt(
            current_velocity.x**2 + current_velocity.y**2 + current_velocity.z**2
        )

        # Get speed limit from CARLA
        target_speed = self._get_carla_speed_limit()
        speed_error = target_speed - current_speed
        throttle = np.clip(self._config.control.throttle_gain * speed_error, 0.0, 1.0)

        control = carla.VehicleControl()
        control.hand_brake = False  # Ensure handbrake is off
        control.throttle = float(throttle)
        control.steer = 0.0
        control.brake = 0.0

        # Check current gear and set to forward if needed
        current_control = self.ego_vehicle.get_control()
        if current_control.gear == 0:
            # If in neutral, manually set gear to 1 (forward)
            control.manual_gear_shift = True
            control.gear = 1
        else:
            # Otherwise use automatic transmission
            control.manual_gear_shift = False

        # Debug output
        print(f"[Control-Simple] Step: {self.step_count:4d} | "
              f"Speed: {current_speed:4.1f}/{target_speed:4.1f} m/s | "
              f"Throttle: {control.throttle:.3f} | "
              f"(No trajectory available)")

        self.ego_vehicle.apply_control(control)

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
            try:
                self.video_writer.release()
                print(f"Videos saved to: {self.log_dir}/ (segments: 0-{self.video_segment_index})")
            except Exception as e:
                print(f"Warning: Error releasing video writer: {e}")
            finally:
                self.video_writer = None

    def __del__(self) -> None:
        """Destructor to ensure video writer is closed."""
        try:
            self.close()
        except Exception:
            # Ignore errors during cleanup
            pass
