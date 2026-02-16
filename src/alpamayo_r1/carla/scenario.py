"""Base scenario class for CARLA simulations."""

from abc import ABC, abstractmethod
from typing import Any


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
    """

    def __init__(
        self,
        world: Any,
        client: Any,
        traffic_manager: Any,
        config: dict | None = None,
        host: str = "localhost",
        port: int = 2000,
    ):
        """Initialize the scenario.

        Args:
            world: CARLA world instance
            client: CARLA client instance
            traffic_manager: CARLA TrafficManager instance
            config: Optional configuration dictionary for scenario parameters
            host: CARLA server host address (for reference/logging)
            port: CARLA server port (for reference/logging)
        """
        self.world = world
        self.client = client
        self.traffic_manager = traffic_manager
        self.ego_vehicle = None
        self.config = config or {}
        self.host = host
        self.port = port

        # Lists to track spawned actors for cleanup
        self.vehicle_npcs = []
        self.pedestrian_npcs = []
        self.pedestrian_controllers = []
        self.sensors = []

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
            if controller is not None and controller.is_alive:
                controller.stop()

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
            if actor is not None and actor.is_alive:
                actor.destroy()

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
