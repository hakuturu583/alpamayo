"""Base scenario class for CARLA simulations."""

from abc import ABC, abstractmethod
from typing import Any

from .controller import AlpamayoController


class BaseScenario(ABC):
    """Base class for CARLA simulation scenarios.

    This class provides a template for creating custom scenarios that can be
    easily integrated with the CARLA simulator. Subclasses should implement
    the abstract methods to define scenario-specific behavior.

    Attributes:
        world: CARLA world instance
        client: CARLA client instance
        traffic_manager: CARLA TrafficManager instance
        ego_vehicle: The ego vehicle actor
        config: Scenario configuration dictionary
        alpamayo_controller: Optional AlpamayoController for model-based control
    """

    def __init__(
        self,
        world: Any,
        client: Any,
        traffic_manager: Any,
        config: dict | None = None,
        host: str = "localhost",
        port: int = 2000,
        model: Any = None,
        processor: Any = None,
        use_alpamayo_control: bool = False,
        use_rerun: bool = False,
    ):
        """Initialize the scenario.

        Args:
            world: CARLA world instance
            client: CARLA client instance
            traffic_manager: CARLA TrafficManager instance
            config: Optional configuration dictionary for scenario parameters
            host: CARLA server host address (for reference/logging)
            port: CARLA server port (for reference/logging)
            model: Optional Alpamayo R1 model for autonomous control
            processor: Optional model processor/tokenizer
            use_alpamayo_control: Whether to use Alpamayo model for ego vehicle control
            use_rerun: Whether to enable Rerun visualization
        """
        self.world = world
        self.client = client
        self.traffic_manager = traffic_manager
        self.ego_vehicle = None
        self.config = config or {}
        self.host = host
        self.port = port

        # Alpamayo controller (initialized after ego vehicle is spawned)
        self.model = model
        self.processor = processor
        self.use_alpamayo_control = use_alpamayo_control
        self.use_rerun = use_rerun
        self.alpamayo_controller = None

        # Lists to track spawned actors for cleanup
        self.vehicle_npcs = []
        self.pedestrian_npcs = []
        self.pedestrian_controllers = []
        self.sensors = []

        # Spawn point management
        self.available_spawn_points = []

    @abstractmethod
    def setup(self) -> None:
        """Set up the scenario.

        This method should:
        - Spawn the ego vehicle
        - Configure scenario-specific settings
        - Set up any initial conditions
        """
        pass

    @abstractmethod
    def spawn_npcs(self) -> None:
        """Spawn NPCs for the scenario.

        This method should:
        - Spawn vehicle NPCs using TrafficManager
        - Spawn pedestrian NPCs with AI walker controllers
        - Configure NPC behaviors
        """
        pass

    @abstractmethod
    def get_spawn_point(self) -> Any:
        """Get the spawn point for the ego vehicle.

        Returns:
            CARLA Transform object for the ego vehicle spawn point
        """
        pass

    def initialize_alpamayo_controller(self, cameras: dict[str, Any]) -> None:
        """Initialize Alpamayo controller after ego vehicle and cameras are ready.

        Args:
            cameras: Dictionary mapping camera names to camera actors
        """
        if self.use_alpamayo_control and self.ego_vehicle is not None:
            self.alpamayo_controller = AlpamayoController(
                ego_vehicle=self.ego_vehicle,
                cameras=cameras,
                model=self.model,
                processor=self.processor,
                use_rerun=self.use_rerun,
                control_frequency=10.0,
            )
            print("Alpamayo controller initialized")

    def update_controller(
        self, world_snapshot: Any, camera_images: dict[str, Any]
    ) -> None:
        """Update Alpamayo controller if enabled.

        Args:
            world_snapshot: CARLA world snapshot
            camera_images: Dictionary mapping camera names to images
        """
        if self.alpamayo_controller is not None:
            self.alpamayo_controller.update(world_snapshot, camera_images)

    def run(self) -> None:
        """Run the scenario.

        This method is called after setup and contains the main scenario logic.
        Override this method to implement custom scenario behavior.
        """
        pass

    def cleanup(self) -> None:
        """Clean up all spawned actors.

        This method destroys all actors spawned during the scenario to prevent
        memory leaks and ensure clean state for subsequent scenarios.
        """
        print("Cleaning up scenario...")

        # Stop pedestrian controllers first
        for controller in self.pedestrian_controllers:
            try:
                if controller is not None and controller.is_alive:
                    controller.stop()
            except RuntimeError:
                pass  # Already destroyed

        # Destroy all actors
        actors_to_destroy = (
            self.sensors
            + self.pedestrian_controllers
            + self.pedestrian_npcs
            + self.vehicle_npcs
        )

        if self.ego_vehicle is not None:
            actors_to_destroy.append(self.ego_vehicle)

        for actor in actors_to_destroy:
            try:
                if actor is not None and actor.is_alive:
                    actor.destroy()
            except RuntimeError:
                pass  # Already destroyed

        # Clear lists
        self.sensors.clear()
        self.pedestrian_controllers.clear()
        self.pedestrian_npcs.clear()
        self.vehicle_npcs.clear()
        self.ego_vehicle = None

        print("Scenario cleanup complete.")

    def on_collision(self, event: Any) -> None:
        """Handle collision events.

        Args:
            event: CARLA collision event
        """
        pass

    def on_invasion(self, event: Any) -> None:
        """Handle lane invasion events.

        Args:
            event: CARLA lane invasion event
        """
        pass

    # Utility methods for common scenario operations

    def allocate_spawn_points(self, ego_spawn_index: int = 0) -> Any:
        """Allocate spawn points, reserving one for ego vehicle.

        Args:
            ego_spawn_index: Index of spawn point to use for ego vehicle

        Returns:
            Spawn point for ego vehicle
        """
        all_spawn_points = self.world.get_map().get_spawn_points()
        print(f"Available spawn points: {len(all_spawn_points)}")

        if len(all_spawn_points) == 0:
            raise RuntimeError("No spawn points available on this map")

        # Get spawn point for ego vehicle
        ego_spawn_index = min(ego_spawn_index, len(all_spawn_points) - 1)
        ego_spawn_point = all_spawn_points[ego_spawn_index]

        # Store remaining spawn points for NPCs
        self.available_spawn_points = [
            sp for i, sp in enumerate(all_spawn_points) if i != ego_spawn_index
        ]

        # Limit vehicle NPCs to available spawn points
        requested_vehicles = self.config.get("num_vehicles", 30)
        max_vehicles = len(self.available_spawn_points)
        if requested_vehicles > max_vehicles:
            print(
                f"Warning: Requested {requested_vehicles} vehicles but only "
                f"{max_vehicles} spawn points available. Limiting to {max_vehicles}."
            )
            self.config["num_vehicles"] = max_vehicles

        return ego_spawn_point

    def spawn_ego_vehicle(
        self, spawn_point: Any = None, vehicle_model: str = "vehicle.tesla.model3"
    ) -> Any:
        """Spawn ego vehicle at specified spawn point with retry logic.

        Args:
            spawn_point: Spawn point for ego vehicle (if None, uses spawn_point_index from config)
            vehicle_model: Vehicle blueprint ID

        Returns:
            Spawned ego vehicle actor
        """
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter(vehicle_model)[0]

        # If spawn_point is provided, try it first
        if spawn_point is not None:
            try:
                self.ego_vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
                print(f"Ego vehicle spawned at {spawn_point.location}")
                return self.ego_vehicle
            except RuntimeError as e:
                print(f"Failed to spawn at provided spawn point: {e}")
                print("Trying alternative spawn points...")

        # Get all spawn points and try them sequentially
        all_spawn_points = self.world.get_map().get_spawn_points()
        if len(all_spawn_points) == 0:
            raise RuntimeError("No spawn points available on this map")

        spawn_idx = self.config.get("spawn_point_index", 0)
        max_attempts = min(10, len(all_spawn_points))  # Try up to 10 spawn points

        for attempt in range(max_attempts):
            try_idx = (spawn_idx + attempt) % len(all_spawn_points)
            try_point = all_spawn_points[try_idx]

            try:
                self.ego_vehicle = self.world.spawn_actor(vehicle_bp, try_point)
                print(f"Ego vehicle spawned at spawn point {try_idx}: {try_point.location}")

                # Update available spawn points (exclude the one we just used)
                self.available_spawn_points = [
                    sp for i, sp in enumerate(all_spawn_points) if i != try_idx
                ]

                # Update vehicle NPC limit
                requested_vehicles = self.config.get("num_vehicles", 30)
                max_vehicles = len(self.available_spawn_points)
                if requested_vehicles > max_vehicles:
                    print(
                        f"Warning: Requested {requested_vehicles} vehicles but only "
                        f"{max_vehicles} spawn points available. Limiting to {max_vehicles}."
                    )
                    self.config["num_vehicles"] = max_vehicles

                return self.ego_vehicle

            except RuntimeError as e:
                if attempt < max_attempts - 1:
                    print(f"Spawn attempt {attempt + 1} failed at point {try_idx}, trying next...")
                else:
                    raise RuntimeError(
                        f"Failed to spawn ego vehicle after {max_attempts} attempts. "
                        f"Last error: {e}"
                    )

        raise RuntimeError("Failed to spawn ego vehicle")

    def spawn_vehicle_npcs(
        self, num_vehicles: int, min_distance_from_ego: float = 20.0
    ) -> list[Any]:
        """Spawn vehicle NPCs using TrafficManager.

        Args:
            num_vehicles: Number of vehicles to spawn
            min_distance_from_ego: Minimum distance from ego vehicle (meters)

        Returns:
            List of spawned vehicle actors
        """
        import numpy as np

        blueprint_library = self.world.get_blueprint_library()
        vehicle_bps = blueprint_library.filter("vehicle.*")

        # Use pre-allocated spawn points
        if not self.available_spawn_points:
            print("Warning: No spawn points available for vehicle NPCs")
            return []

        # Filter spawn points by distance from ego vehicle
        ego_location = self.ego_vehicle.get_location() if self.ego_vehicle else None
        valid_spawn_points = []

        for spawn_point in self.available_spawn_points:
            if ego_location is None:
                valid_spawn_points.append(spawn_point)
            else:
                distance = spawn_point.location.distance(ego_location)
                if distance >= min_distance_from_ego:
                    valid_spawn_points.append(spawn_point)

        if not valid_spawn_points:
            print(
                f"Warning: No spawn points found with minimum distance "
                f"{min_distance_from_ego}m from ego vehicle"
            )
            return []

        print(
            f"Found {len(valid_spawn_points)} valid spawn points "
            f"(min distance: {min_distance_from_ego}m)"
        )

        # Limit to valid spawn points
        num_to_spawn = min(num_vehicles, len(valid_spawn_points))

        spawned = 0
        for i in range(num_to_spawn):
            spawn_point = valid_spawn_points[i]
            vehicle_bp = np.random.choice(vehicle_bps)

            try:
                # Use try_spawn_actor for safer spawning
                vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)
                if vehicle is not None:
                    vehicle.set_autopilot(True, self.traffic_manager.get_port())
                    self.vehicle_npcs.append(vehicle)
                    spawned += 1
                else:
                    # try_spawn_actor returns None on failure (no exception)
                    pass
            except Exception as e:
                print(f"Failed to spawn vehicle at point {i}: {e}")
                continue

        print(f"Spawned {len(self.vehicle_npcs)} vehicle NPCs (requested: {num_vehicles})")
        return self.vehicle_npcs

    def spawn_pedestrian_npcs(
        self, num_pedestrians: int, spawn_radius: float = 60.0
    ) -> tuple[list[Any], list[Any]]:
        """Spawn pedestrian NPCs with AI walker controllers.

        Args:
            num_pedestrians: Number of pedestrians to spawn
            spawn_radius: Radius around ego vehicle for spawning (in meters)

        Returns:
            Tuple of (pedestrian actors, controller actors)
        """
        import carla
        import numpy as np
        import time

        if self.ego_vehicle is None:
            print("Warning: Ego vehicle must be spawned before pedestrians")
            return [], []

        blueprint_library = self.world.get_blueprint_library()
        walker_bps = blueprint_library.filter("walker.pedestrian.*")
        controller_bp = blueprint_library.find("controller.ai.walker")

        ego_location = self.ego_vehicle.get_location()

        print(f"Spawning {num_pedestrians} pedestrians...")

        # Spawn pedestrians with safety checks
        spawn_batch = []
        for i in range(num_pedestrians):
            spawn_transform = carla.Transform()
            spawn_transform.location = carla.Location(
                x=ego_location.x + np.random.uniform(-spawn_radius, spawn_radius),
                y=ego_location.y + np.random.uniform(-spawn_radius, spawn_radius),
                z=ego_location.z + 1.0,  # Slightly higher to avoid ground collision
            )

            walker_bp = np.random.choice(walker_bps)

            try:
                pedestrian = self.world.try_spawn_actor(walker_bp, spawn_transform)
                if pedestrian is not None:
                    self.pedestrian_npcs.append(pedestrian)
                    spawn_batch.append(pedestrian)

                    # Tick every few spawns to help CARLA process
                    if (i + 1) % 5 == 0:
                        self.world.tick()

            except Exception as e:
                print(f"Failed to spawn pedestrian {i}: {e}")
                continue

        # Final tick to register all pedestrians
        self.world.tick()
        time.sleep(0.1)  # Small delay for stability

        if len(self.pedestrian_npcs) == 0:
            print("No pedestrians spawned")
            return [], []

        print(f"Spawned {len(self.pedestrian_npcs)} pedestrians, creating controllers...")

        # Spawn controllers with safety checks
        for i, pedestrian in enumerate(self.pedestrian_npcs):
            try:
                if pedestrian is None or not pedestrian.is_alive:
                    continue

                controller = self.world.try_spawn_actor(
                    controller_bp, carla.Transform(), pedestrian
                )
                if controller is not None:
                    self.pedestrian_controllers.append(controller)

                    # Tick every few spawns
                    if (i + 1) % 5 == 0:
                        self.world.tick()

            except Exception as e:
                print(f"Failed to spawn controller for pedestrian {i}: {e}")
                continue

        # Final tick to register all controllers
        self.world.tick()
        time.sleep(0.1)  # Small delay for stability

        print(f"Created {len(self.pedestrian_controllers)} controllers, starting AI...")

        # Start controllers with minimal operations to avoid segfaults
        # AI walkers will automatically wander randomly without explicit destinations
        for i, controller in enumerate(self.pedestrian_controllers):
            try:
                if controller is None or not controller.is_alive:
                    continue

                # Only start the controller - let AI handle movement automatically
                controller.start()

                # Tick after EVERY start for maximum stability
                self.world.tick()
                time.sleep(0.1)  # Longer delay for stability

            except Exception as e:
                print(f"Failed to start controller {i}: {e}")
                continue

        # Final wait to ensure all controllers are fully started
        self.world.tick()
        time.sleep(0.3)

        print(f"Successfully spawned {len(self.pedestrian_npcs)} pedestrian NPCs with {len(self.pedestrian_controllers)} controllers")
        print("Pedestrians will wander randomly (no explicit destinations set)")
        return self.pedestrian_npcs, self.pedestrian_controllers
