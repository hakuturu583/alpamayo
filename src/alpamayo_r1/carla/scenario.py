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
        """Spawn ego vehicle at specified spawn point.

        Args:
            spawn_point: Spawn point for ego vehicle (if None, uses spawn_point_index from config)
            vehicle_model: Vehicle blueprint ID

        Returns:
            Spawned ego vehicle actor
        """
        if spawn_point is None:
            spawn_idx = self.config.get("spawn_point_index", 0)
            spawn_point = self.allocate_spawn_points(spawn_idx)

        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter(vehicle_model)[0]
        self.ego_vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)

        print(f"Ego vehicle spawned at {spawn_point.location}")
        return self.ego_vehicle

    def spawn_vehicle_npcs(self, num_vehicles: int) -> list[Any]:
        """Spawn vehicle NPCs using TrafficManager.

        Args:
            num_vehicles: Number of vehicles to spawn

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

        # Limit to available spawn points
        num_to_spawn = min(num_vehicles, len(self.available_spawn_points))

        spawned = 0
        for i in range(num_to_spawn):
            spawn_point = self.available_spawn_points[i]
            vehicle_bp = np.random.choice(vehicle_bps)

            try:
                vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
                vehicle.set_autopilot(True, self.traffic_manager.get_port())
                self.vehicle_npcs.append(vehicle)
                spawned += 1
            except RuntimeError as e:
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
        import numpy as np

        if self.ego_vehicle is None:
            print("Warning: Ego vehicle must be spawned before pedestrians")
            return [], []

        blueprint_library = self.world.get_blueprint_library()
        walker_bps = blueprint_library.filter("walker.pedestrian.*")
        controller_bp = blueprint_library.find("controller.ai.walker")

        ego_location = self.ego_vehicle.get_location()

        # Spawn pedestrians
        for _ in range(num_pedestrians):
            spawn_point = self.world.get_blueprint_library().find("controller.ai.walker")
            spawn_point = type("Transform", (), {})()
            spawn_point.location = ego_location + type("Location", (), {})(
                x=np.random.uniform(-spawn_radius, spawn_radius),
                y=np.random.uniform(-spawn_radius, spawn_radius),
                z=0.5,
            )

            # Convert to proper CARLA Transform
            import carla

            spawn_transform = carla.Transform()
            spawn_transform.location = carla.Location(
                x=ego_location.x + np.random.uniform(-spawn_radius, spawn_radius),
                y=ego_location.y + np.random.uniform(-spawn_radius, spawn_radius),
                z=ego_location.z + 0.5,
            )

            walker_bp = np.random.choice(walker_bps)

            try:
                pedestrian = self.world.spawn_actor(walker_bp, spawn_transform)
                self.pedestrian_npcs.append(pedestrian)
            except RuntimeError:
                continue

        # Wait for pedestrians to be registered
        self.world.tick()

        # Spawn controllers
        for pedestrian in self.pedestrian_npcs:
            try:
                import carla

                controller = self.world.spawn_actor(
                    controller_bp, carla.Transform(), pedestrian
                )
                self.pedestrian_controllers.append(controller)
            except RuntimeError:
                continue

        # Wait for controllers to be registered
        self.world.tick()

        # Start walking behavior
        for controller in self.pedestrian_controllers:
            import carla

            controller.start()
            destination = carla.Location(
                x=ego_location.x + np.random.uniform(-spawn_radius, spawn_radius),
                y=ego_location.y + np.random.uniform(-spawn_radius, spawn_radius),
                z=ego_location.z,
            )
            controller.go_to_location(destination)
            controller.set_max_speed(np.random.uniform(1.0, 2.0))

        print(f"Spawned {len(self.pedestrian_npcs)} pedestrian NPCs")
        return self.pedestrian_npcs, self.pedestrian_controllers
