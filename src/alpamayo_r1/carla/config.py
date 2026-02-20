"""Configuration dataclasses for the CARLA simulation pipeline.

All tuneable parameters are gathered here and can be loaded from a YAML file:

    cfg = CarlaConfig.from_yaml("config/carla_default.yaml")

Unknown YAML keys are silently ignored so that partial override files work
without needing to spell out every default.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import yaml


@dataclass
class CameraParams:
    """Camera hardware layout and image resolution."""

    width: int = 1920
    height: int = 1080
    height_offset: float = 1.5           # z above vehicle actor origin [m]
    cross_lateral_offset: float = 1.2    # |y| for side cameras [m]
    front_longitudinal_offset: float = 2.0  # x for front cameras [m]
    front_wide_fov: float = 120.0        # front wide-angle FOV [deg]
    cross_fov: float = 120.0             # side cross-camera FOV [deg]
    front_tele_fov: float = 30.0         # front telephoto FOV [deg]


@dataclass
class ControlParams:
    """Vehicle control and trajectory-following parameters."""

    max_speed: float = 15.0              # hard speed cap [m/s]
    lateral_lookahead: float = 15.0      # fixed lateral lookahead distance [m]
    min_lookahead_distance: float = 4.5  # minimum lookahead distance [m]
    spline_num_points: int = 200         # spline interpolation resolution
    speed_limit_factor: float = 0.8      # fraction of road speed limit to target
    max_lateral_accel: float = 4.0       # max lateral accel for curve speed limit [m/s²]
    throttle_gain: float = 0.5           # P-gain for speed error → throttle
    brake_threshold: float = -0.5        # speed error [m/s] to engage braking
    brake_value: float = 0.3             # normalised brake command
    speed_reduction_threshold: float = 0.75  # trajectory length ratio to start slowing
    min_speed_factor: float = 0.1        # lower bound on speed reduction multiplier
    rear_axle_offset: float = 0.5        # rear bumper → rear axle estimate [m]
    steering_alpha: float = 0.3          # EMA smoothing factor for steering (0=max smooth, 1=no smooth)
    target_speed_alpha: float = 0.4      # EMA smoothing factor for target speed (0=max smooth, 1=no smooth)
    curvature_gain: float = 1.0          # scale factor on curvature command (<1 reduces inside-corner bias)


@dataclass
class InferenceParams:
    """Model sampling and frame-buffer parameters."""

    top_p: float = 0.98
    temperature: float = 0.6
    num_traj_samples: int = 1
    max_generation_length: int = 256
    num_frame_history: int = 4           # rolling camera frames kept per camera
    control_frequency: float = 10.0      # inference / control update rate [Hz]


@dataclass
class SimulationParams:
    """CARLA connection and scenario parameters."""

    host: str = "localhost"
    port: int = 2000
    timeout: float = 10.0                # CARLA client timeout [s]
    delta_seconds: float = 0.05          # fixed simulation timestep (= 20 Hz)
    map_name: str = "Town01"
    num_vehicles: int = 50
    num_pedestrians: int = 30
    num_steps: int = 2000
    spawn_point_index: int = 0
    seed: int = 0
    vehicle_model: str = "vehicle.tesla.model3"
    traffic_manager_port: int = 8000
    min_vehicle_distance: float = 20.0   # min spawn distance from ego [m]
    pedestrian_spawn_radius: float = 60.0
    pedestrian_speed_min: float = 1.0    # [m/s]
    pedestrian_speed_max: float = 2.0    # [m/s]


@dataclass
class VideoParams:
    """Visualization video recording parameters."""

    save_video: bool = True
    frames_per_segment: int = 100        # frames before starting a new video file
    fps: float = 20.0


@dataclass
class CarlaConfig:
    """Root configuration for the full CARLA pipeline."""

    camera: CameraParams = field(default_factory=CameraParams)
    control: ControlParams = field(default_factory=ControlParams)
    inference: InferenceParams = field(default_factory=InferenceParams)
    simulation: SimulationParams = field(default_factory=SimulationParams)
    video: VideoParams = field(default_factory=VideoParams)

    @classmethod
    def from_yaml(cls, path: str) -> "CarlaConfig":
        """Load from a YAML file.  Unknown keys are silently ignored."""
        with open(path) as f:
            data = yaml.safe_load(f) or {}

        def _load(dc_cls, section: str):
            raw = data.get(section, {})
            known = {f.name for f in dataclasses.fields(dc_cls)}
            return dc_cls(**{k: v for k, v in raw.items() if k in known})

        return cls(
            camera=_load(CameraParams, "camera"),
            control=_load(ControlParams, "control"),
            inference=_load(InferenceParams, "inference"),
            simulation=_load(SimulationParams, "simulation"),
            video=_load(VideoParams, "video"),
        )
