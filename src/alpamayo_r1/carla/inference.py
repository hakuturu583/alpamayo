"""CARLA simulation integration for Alpamayo R1 inference.

This module provides integration with CARLA simulator for autonomous vehicle
simulation using the Alpamayo R1 model. The camera setup mimics the NVIDIA
PhysicalAI Autonomous Vehicles dataset configuration.
"""

import queue
import time
from typing import Any

import carla
import numpy as np
from tqdm import tqdm

from .config import CarlaConfig, CameraParams
from .scenario import BaseScenario


class CameraConfig:
    """Camera configuration matching PhysicalAI-AV dataset.

    The coordinate frame origin is at the center of the rear axle, projected
    onto the ground plane:
    - X-axis: points forward
    - Y-axis: points left (when looking forward)
    - Z-axis: points up

    Build from a CameraParams dataclass so that positions and FOVs are
    configurable without touching this file.
    """

    def __init__(self, params: CameraParams | None = None) -> None:
        if params is None:
            params = CameraParams()

        self.WIDTH = params.width
        self.HEIGHT = params.height

        z = params.height_offset
        lx = params.front_longitudinal_offset
        ly = params.cross_lateral_offset

        self.CAMERAS = [
            {
                "name": "camera_cross_left_120fov",
                "index": 0,
                "fov": params.cross_fov,
                "transform": carla.Transform(
                    carla.Location(x=0.0, y=-ly, z=z),
                    carla.Rotation(pitch=0.0, yaw=-90.0, roll=0.0),
                ),
            },
            {
                "name": "camera_front_wide_120fov",
                "index": 1,
                "fov": params.front_wide_fov,
                "transform": carla.Transform(
                    carla.Location(x=lx, y=0.0, z=z),
                    carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0),
                ),
            },
            {
                "name": "camera_cross_right_120fov",
                "index": 2,
                "fov": params.cross_fov,
                "transform": carla.Transform(
                    carla.Location(x=0.0, y=ly, z=z),
                    carla.Rotation(pitch=0.0, yaw=90.0, roll=0.0),
                ),
            },
            {
                "name": "camera_front_tele_30fov",
                "index": 6,
                "fov": params.front_tele_fov,
                "transform": carla.Transform(
                    carla.Location(x=lx, y=0.0, z=z),
                    carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0),
                ),
            },
        ]


class CARLASimulation:
    """Main class for CARLA simulation with Alpamayo R1 integration.

    This class manages the CARLA connection, spawns cameras matching the
    PhysicalAI-AV dataset configuration, and provides interfaces for running
    scenarios with vehicle and pedestrian NPCs.
    """

    def __init__(self, config: CarlaConfig | None = None) -> None:
        """Initialize CARLA simulation.

        Args:
            config: Full pipeline configuration.  Defaults to CarlaConfig()
                    (all dataclass defaults) when not provided.
        """
        self._config = config or CarlaConfig()
        sim = self._config.simulation

        self.host = sim.host
        self.port = sim.port
        self.timeout = sim.timeout

        # CARLA client and world
        self.client = None
        self.world = None
        self.traffic_manager = None

        # Ego vehicle and sensors
        self.ego_vehicle = None
        self.cameras = {}
        self.camera_queues = {}

        # Map
        self.map_name = sim.map_name

        # Camera config built from params
        self._camera_config = CameraConfig(self._config.camera)

        print(f"Initializing CARLA simulation at {self.host}:{self.port}...")

    def connect(self) -> None:
        """Connect to CARLA server and initialize world."""
        self.client = carla.Client(self.host, self.port)
        self.client.set_timeout(self.timeout)

        # Load map if specified
        if self.map_name is not None:
            print(f"Loading map: {self.map_name}")
            self.world = self.client.load_world(self.map_name)
        else:
            self.world = self.client.get_world()

        # Clean up all existing actors from previous sessions
        print("Cleaning up existing actors...")
        self._cleanup_all_actors()

        # Initialize TrafficManager
        sim = self._config.simulation
        self.traffic_manager = self.client.get_trafficmanager(sim.traffic_manager_port)
        self.traffic_manager.set_synchronous_mode(True)

        # Set synchronous mode for deterministic simulation
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = sim.delta_seconds
        self.world.apply_settings(settings)

        print(f"Connected to CARLA. Map: {self.world.get_map().name}")

    def spawn_ego_vehicle(
        self, spawn_point: carla.Transform | None = None, vehicle_model: str = "vehicle.tesla.model3"
    ) -> carla.Actor:
        """Spawn the ego vehicle.

        Args:
            spawn_point: Optional spawn point. If None, uses a random spawn point.
            vehicle_model: Vehicle blueprint ID

        Returns:
            The spawned ego vehicle actor
        """
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter(vehicle_model)[0]

        if spawn_point is None:
            spawn_points = self.world.get_map().get_spawn_points()
            spawn_point = spawn_points[0] if spawn_points else carla.Transform()

        self.ego_vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        print(f"Spawned ego vehicle at {spawn_point.location}")

        return self.ego_vehicle

    def setup_cameras(self) -> dict[str, carla.Actor]:
        """Set up cameras matching PhysicalAI-AV dataset configuration.

        Returns:
            Dictionary mapping camera names to camera actors
        """
        if self.ego_vehicle is None:
            raise RuntimeError("Ego vehicle must be spawned before setting up cameras")

        blueprint_library = self.world.get_blueprint_library()
        camera_bp = blueprint_library.find("sensor.camera.rgb")

        # Configure camera resolution
        camera_bp.set_attribute("image_size_x", str(self._camera_config.WIDTH))
        camera_bp.set_attribute("image_size_y", str(self._camera_config.HEIGHT))

        # Spawn all cameras
        for cam_config in self._camera_config.CAMERAS:
            # Set FOV for this camera
            camera_bp.set_attribute("fov", str(cam_config["fov"]))

            # Spawn camera attached to ego vehicle
            camera = self.world.spawn_actor(
                camera_bp,
                cam_config["transform"],
                attach_to=self.ego_vehicle,
                attachment_type=carla.AttachmentType.Rigid,
            )

            # Set up image queue for this camera
            image_queue = queue.Queue()
            camera.listen(lambda image, q=image_queue: q.put(image))

            self.cameras[cam_config["name"]] = camera
            self.camera_queues[cam_config["name"]] = image_queue

            print(f"Spawned camera: {cam_config['name']} (FOV: {cam_config['fov']}°)")

        return self.cameras

    def spawn_vehicle_npcs(
        self, num_vehicles: int = 50, safe_distance: float = 10.0
    ) -> list[carla.Actor]:
        """Spawn vehicle NPCs controlled by TrafficManager.

        Args:
            num_vehicles: Number of vehicles to spawn
            safe_distance: Minimum distance from ego vehicle for spawning

        Returns:
            List of spawned vehicle actors
        """
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bps = blueprint_library.filter("vehicle.*")

        spawn_points = self.world.get_map().get_spawn_points()
        vehicles = []

        # Get ego vehicle location for safe distance check
        ego_location = self.ego_vehicle.get_location() if self.ego_vehicle else None

        for i in range(num_vehicles):
            if i >= len(spawn_points):
                break

            spawn_point = spawn_points[i]

            # Check safe distance from ego vehicle
            if ego_location is not None:
                distance = spawn_point.location.distance(ego_location)
                if distance < safe_distance:
                    continue

            # Randomly select vehicle blueprint
            vehicle_bp = np.random.choice(vehicle_bps)

            # Spawn vehicle
            try:
                vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
                vehicle.set_autopilot(True, self.traffic_manager.get_port())
                vehicles.append(vehicle)
            except RuntimeError as e:
                print(f"Failed to spawn vehicle at point {i}: {e}")
                continue

        print(f"Spawned {len(vehicles)} vehicle NPCs with TrafficManager")
        return vehicles

    def spawn_pedestrian_npcs(
        self, num_pedestrians: int = 30, spawn_radius: float = 50.0
    ) -> tuple[list[carla.Actor], list[carla.Actor]]:
        """Spawn pedestrian NPCs with AI walker controllers.

        Args:
            num_pedestrians: Number of pedestrians to spawn
            spawn_radius: Radius around ego vehicle for spawning pedestrians

        Returns:
            Tuple of (pedestrian actors, controller actors)
        """
        blueprint_library = self.world.get_blueprint_library()
        walker_bps = blueprint_library.filter("walker.pedestrian.*")
        controller_bp = blueprint_library.find("controller.ai.walker")

        pedestrians = []
        controllers = []

        # Get spawn locations near ego vehicle
        if self.ego_vehicle is not None:
            ego_location = self.ego_vehicle.get_location()
        else:
            spawn_points = self.world.get_map().get_spawn_points()
            ego_location = spawn_points[0].location if spawn_points else carla.Location()

        # Spawn pedestrians
        for _ in range(num_pedestrians):
            # Random spawn location within radius
            spawn_point = carla.Transform()
            spawn_point.location = ego_location + carla.Location(
                x=np.random.uniform(-spawn_radius, spawn_radius),
                y=np.random.uniform(-spawn_radius, spawn_radius),
                z=0.5,  # Slightly above ground
            )

            walker_bp = np.random.choice(walker_bps)

            try:
                pedestrian = self.world.spawn_actor(walker_bp, spawn_point)
                pedestrians.append(pedestrian)
            except RuntimeError as e:
                print(f"Failed to spawn pedestrian: {e}")
                continue

        # Wait a tick for pedestrians to be properly registered
        self.world.tick()

        # Spawn controllers for each pedestrian
        for pedestrian in pedestrians:
            try:
                controller = self.world.spawn_actor(controller_bp, carla.Transform(), pedestrian)
                controllers.append(controller)
            except RuntimeError as e:
                print(f"Failed to spawn controller: {e}")
                continue

        # Wait another tick
        self.world.tick()

        # Start walking behavior for all pedestrians
        for controller in controllers:
            controller.start()
            # Set random destination
            controller.go_to_location(
                ego_location
                + carla.Location(
                    x=np.random.uniform(-spawn_radius, spawn_radius),
                    y=np.random.uniform(-spawn_radius, spawn_radius),
                    z=0.0,
                )
            )
            # Set random speed (1-2 m/s)
            controller.set_max_speed(np.random.uniform(1.0, 2.0))

        print(f"Spawned {len(pedestrians)} pedestrian NPCs with AI walker controllers")
        return pedestrians, controllers

    def get_camera_images(self, timeout: float = 1.0) -> dict[str, np.ndarray]:
        """Get latest images from all cameras.

        Args:
            timeout: Maximum time to wait for images

        Returns:
            Dictionary mapping camera names to image arrays (H, W, 3) in RGB format
        """
        images = {}

        for name, image_queue in self.camera_queues.items():
            try:
                # Get latest image and discard old ones to prevent memory buildup
                image = None
                while True:
                    try:
                        image = image_queue.get(block=False)
                    except queue.Empty:
                        break

                # If no image in queue, wait for new one
                if image is None:
                    image = image_queue.get(timeout=timeout)

                # Convert CARLA image to numpy array (BGRA -> RGB)
                array = np.frombuffer(image.raw_data, dtype=np.uint8)
                array = array.reshape((image.height, image.width, 4))
                array = array[:, :, :3]  # Remove alpha channel
                array = array[:, :, ::-1]  # BGR to RGB

                images[name] = array
            except queue.Empty:
                print(f"Warning: Timeout waiting for image from {name}")

        return images

    def run_scenario(self, scenario: BaseScenario, num_steps: int = 1000) -> None:
        """Run a scenario in the simulation.

        Args:
            scenario: Scenario instance to run
            num_steps: Number of simulation steps to run
        """
        try:
            print(f"Setting up scenario: {scenario.__class__.__name__}")
            scenario.setup()

            # Set ego_vehicle reference from scenario
            self.ego_vehicle = scenario.ego_vehicle

            # Setup cameras after ego vehicle is spawned
            self.setup_cameras()

            # Initialize Alpamayo controller if model-based control is enabled
            if scenario.use_alpamayo_control:
                print("Initializing Alpamayo controller...")
                save_video = scenario._config.video.save_video
                scenario.initialize_alpamayo_controller(self.cameras, save_video=save_video)

            print("Spawning NPCs...")
            scenario.spawn_npcs()

            print(f"Running scenario for {num_steps} steps...")
            for step in tqdm(range(num_steps), desc="Simulation", unit="step"):
                try:
                    # Tick the world
                    self.world.tick()
                    snapshot = self.world.get_snapshot()

                    # Get camera images
                    images = self.get_camera_images()
                    if len(images) == 0:
                        print(f"Warning: No camera images received at step {step}")

                    # Update Alpamayo controller if enabled
                    if scenario.use_alpamayo_control:
                        scenario.update_controller(snapshot, images)

                    # Run scenario-specific logic
                    scenario.run()

                except KeyboardInterrupt:
                    print("\nSimulation interrupted by user")
                    break
                except Exception as e:
                    print(f"\nError at step {step}: {e}")
                    import traceback
                    traceback.print_exc()
                    break

        finally:
            print("Cleaning up scenario...")
            scenario.cleanup()

    def _cleanup_all_actors(self) -> None:
        """Clean up all actors in the world (vehicles, walkers, sensors).

        This is called at initialization to ensure a clean state.
        """
        if self.world is None:
            return

        actors = self.world.get_actors()

        # Categorize actors
        vehicles = []
        walkers = []
        sensors = []
        controllers = []

        for actor in actors:
            type_id = actor.type_id
            if type_id.startswith("vehicle."):
                vehicles.append(actor)
            elif type_id.startswith("walker.pedestrian."):
                walkers.append(actor)
            elif type_id.startswith("sensor."):
                sensors.append(actor)
            elif type_id.startswith("controller.ai.walker"):
                controllers.append(actor)

        # Destroy in order: controllers first, then sensors, then walkers, then vehicles
        destroyed_count = 0

        for controller in controllers:
            try:
                if controller.is_alive:
                    controller.stop()
                    controller.destroy()
                    destroyed_count += 1
            except Exception:
                pass

        for sensor in sensors:
            try:
                if sensor.is_alive:
                    sensor.destroy()
                    destroyed_count += 1
            except Exception:
                pass

        for walker in walkers:
            try:
                if walker.is_alive:
                    walker.destroy()
                    destroyed_count += 1
            except Exception:
                pass

        for vehicle in vehicles:
            try:
                if vehicle.is_alive:
                    vehicle.destroy()
                    destroyed_count += 1
            except Exception:
                pass

        if destroyed_count > 0:
            print(f"Destroyed {destroyed_count} existing actors")

        # Tick to process deletions
        try:
            self.world.tick()
        except Exception:
            pass

    def cleanup(self) -> None:
        """Clean up all actors and restore settings."""
        print("Cleaning up CARLA simulation...")

        # NOTE: Don't touch cameras, ego_vehicle, or world settings here
        # scenario.cleanup() has already reloaded the map, which:
        # 1. Destroys all actors automatically
        # 2. Creates a new world object (making self.world invalid)
        # 3. Resets all settings to default
        # Any attempt to access self.world here will fail with "destroyed actor" error

        # Clear references only (actors are already cleaned up by map reload)
        self.cameras.clear()
        self.camera_queues.clear()
        self.ego_vehicle = None

        print("Cleanup complete.")

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()
