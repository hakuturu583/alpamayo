"""Decoupled lateral and longitudinal controller for autonomous driving.

This module implements a decoupled waypoint representation approach where
lateral (steering) and longitudinal (speed) controls are computed independently:

- Lateral control: Fixed lookahead distance on spline-interpolated trajectory
- Longitudinal control: Speed control based on target speed

This decoupling prevents issues where speed reduction causes lookahead distance
to shrink, leading to delayed steering response in curves.
"""

import numpy as np
from typing import Tuple, Optional


class DecoupledController:
    """Decoupled lateral and longitudinal controller.

    Attributes:
        wheelbase: Vehicle wheelbase in meters
        max_steering: Maximum steering angle in radians
        lateral_lookahead: Fixed lookahead distance for lateral control (meters)
        min_lookahead_distance: Minimum lookahead distance (meters)
        spline_num_points: Number of points for spline interpolation
    """

    def __init__(
        self,
        wheelbase: float,
        max_steering: float,
        lateral_lookahead: float = 15.0,
        min_lookahead_distance: float = 4.5,
        spline_num_points: int = 200,
        throttle_gain: float = 0.5,
        brake_threshold: float = -0.5,
        brake_value: float = 0.3,
    ):
        """Initialize decoupled controller.

        Args:
            wheelbase: Vehicle wheelbase in meters
            max_steering: Maximum steering angle in radians
            lateral_lookahead: Fixed lookahead distance for lateral control (default: 15.0m)
            min_lookahead_distance: Minimum lookahead distance (default: 4.5m)
            spline_num_points: Number of points for spline interpolation (default: 200)
            throttle_gain: P-gain for speed error to throttle mapping (default: 0.5)
            brake_threshold: Speed error [m/s] below which braking is applied (default: -0.5)
            brake_value: Normalised brake command when braking (default: 0.3)
        """
        self.wheelbase = wheelbase
        self.max_steering = max_steering
        self.lateral_lookahead = lateral_lookahead
        self.min_lookahead_distance = min_lookahead_distance
        self.spline_num_points = spline_num_points
        self.throttle_gain = throttle_gain
        self.brake_threshold = brake_threshold
        self.brake_value = brake_value

    def interpolate_trajectory_spline(
        self, trajectory: np.ndarray, num_points: Optional[int] = None
    ) -> np.ndarray:
        """Interpolate trajectory using cubic spline for smooth path following.

        Args:
            trajectory: (N, 3) array [x, y, z]
            num_points: Number of interpolated points (default: use self.spline_num_points)

        Returns:
            Interpolated trajectory (num_points, 3) [x, y, z]
        """
        from scipy.interpolate import splprep, splev

        if num_points is None:
            num_points = self.spline_num_points

        if len(trajectory) < 4:
            # Not enough points for spline, return original
            return trajectory

        # Extract x, y (ignore z for now)
        x = trajectory[:, 0]
        y = trajectory[:, 1]

        try:
            # Fit cubic spline (k=3, s=0 for exact interpolation)
            k = min(3, len(x) - 1)
            tck, u = splprep([x, y], s=0, k=k)

            # Evaluate spline at uniform parameter values
            u_new = np.linspace(0, 1, num_points)
            x_new, y_new = splev(u_new, tck)

            # Keep z=0 for 2D trajectory
            z_new = np.zeros_like(x_new)

            return np.column_stack([x_new, y_new, z_new])

        except Exception as e:
            # Fallback to original trajectory if spline fails
            print(f"[Spline Warning] Failed to interpolate trajectory: {e}")
            return trajectory

    def get_lateral_target_point(
        self, trajectory_dense: np.ndarray, lookahead_distance: Optional[float] = None
    ) -> Tuple[np.ndarray, int, float]:
        """Get lateral control target point at fixed lookahead distance.

        Args:
            trajectory_dense: Dense interpolated trajectory (N, 3) [x, y, z]
            lookahead_distance: Lookahead distance (default: use self.lateral_lookahead)

        Returns:
            Tuple of (target_point (3,), waypoint_index, actual_distance)
        """
        if lookahead_distance is None:
            lookahead_distance = self.lateral_lookahead

        # Calculate cumulative distance along trajectory
        positions = trajectory_dense[:, :2]  # (N, 2)
        deltas = np.diff(positions, axis=0)
        distances = np.linalg.norm(deltas, axis=1)
        cumulative_distances = np.concatenate([[0], np.cumsum(distances)])

        # Find point closest to lookahead distance
        # Limit to trajectory range
        max_distance = cumulative_distances[-1]
        effective_lookahead = min(lookahead_distance, max_distance * 0.95)

        # Ensure minimum lookahead distance
        effective_lookahead = max(effective_lookahead, self.min_lookahead_distance)

        idx = np.argmin(np.abs(cumulative_distances - effective_lookahead))

        # Clamp to valid range
        idx = np.clip(idx, 0, len(trajectory_dense) - 1)

        return trajectory_dense[idx], idx, cumulative_distances[idx]

    def compute_lateral_control(
        self, trajectory: np.ndarray
    ) -> Tuple[float, float, np.ndarray, int, float]:
        """Compute lateral control (steering) using Pure Pursuit on spline-interpolated trajectory.

        Args:
            trajectory: (N, 3) array [x, y, z] in vehicle local frame

        Returns:
            Tuple of:
                - steering: Normalized steering command [-1.0, 1.0]
                - steering_angle_rad: Steering angle in radians
                - target_point: Target point (3,) [x, y, z]
                - target_idx: Index of target point in dense trajectory
                - lookahead_distance: Actual lookahead distance used
        """
        if len(trajectory) < 2:
            # Not enough points, go straight
            return 0.0, 0.0, np.array([0.0, 0.0, 0.0]), 0, 0.0

        # Step 1: Interpolate trajectory with spline
        trajectory_dense = self.interpolate_trajectory_spline(trajectory)

        # Step 2: Get lateral target point at fixed lookahead distance
        target_point, target_idx, actual_lookahead = self.get_lateral_target_point(
            trajectory_dense
        )

        target_x = target_point[0]
        target_y = target_point[1]
        lookahead_distance = np.sqrt(target_x**2 + target_y**2)

        # Step 3: Pure pursuit steering calculation
        steering_angle_rad = 0.0
        steering = 0.0

        if lookahead_distance > 0.1:
            # Curvature = 2 * lateral_error / lookahead_distance^2
            # Model coords: Y=left (positive = target on left)
            # CARLA control: steer positive = turn left
            curvature = 2.0 * target_y / (lookahead_distance**2)

            # Convert curvature to steering angle
            # steering_angle = atan(wheelbase * curvature)
            steering_angle_rad = np.arctan(self.wheelbase * curvature)

            # Clamp to max steering angle
            steering_angle_rad = np.clip(
                steering_angle_rad, -self.max_steering, self.max_steering
            )

            # Normalize to CARLA control range [-1.0, 1.0]
            steering = steering_angle_rad / self.max_steering

        return steering, steering_angle_rad, target_point, target_idx, actual_lookahead

    def compute_longitudinal_control(
        self, target_speed: float, current_speed: float
    ) -> Tuple[float, float]:
        """Compute longitudinal control (throttle/brake) based on speed error.

        Args:
            target_speed: Target speed in m/s
            current_speed: Current speed in m/s

        Returns:
            Tuple of (throttle, brake) both in [0.0, 1.0]
        """
        speed_error = target_speed - current_speed

        # Proportional speed control
        throttle = np.clip(self.throttle_gain * speed_error, 0.0, 1.0)

        # Apply brake if significantly over target speed
        brake = 0.0 if speed_error > self.brake_threshold else self.brake_value

        return throttle, brake

    def compute_control(
        self, trajectory: np.ndarray, target_speed: float, current_speed: float
    ) -> dict:
        """Compute decoupled lateral and longitudinal control commands.

        Args:
            trajectory: (N, 3) array [x, y, z] in vehicle local frame
            target_speed: Target speed in m/s
            current_speed: Current speed in m/s

        Returns:
            Dictionary containing:
                - steering: Normalized steering [-1.0, 1.0]
                - throttle: Throttle [0.0, 1.0]
                - brake: Brake [0.0, 1.0]
                - steering_angle_rad: Steering angle in radians
                - target_point: Target point (3,) [x, y, z]
                - target_idx: Waypoint index in dense trajectory
                - lookahead_distance: Actual lookahead distance
        """
        # Lateral control
        steering, steering_angle_rad, target_point, target_idx, lookahead_distance = (
            self.compute_lateral_control(trajectory)
        )

        # Longitudinal control
        throttle, brake = self.compute_longitudinal_control(target_speed, current_speed)

        return {
            "steering": steering,
            "throttle": throttle,
            "brake": brake,
            "steering_angle_rad": steering_angle_rad,
            "target_point": target_point,
            "target_idx": target_idx,
            "lookahead_distance": lookahead_distance,
        }
