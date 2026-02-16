"""Example scenario demonstrating how to use the CARLA simulation framework."""

import carla

from .inference import CARLASimulation
from .scenario import BaseScenario


class SimpleUrbanScenario(BaseScenario):
    """Simple urban driving scenario with vehicle and pedestrian NPCs.

    This scenario demonstrates:
    - Spawning ego vehicle at a specific location
    - Spawning vehicle NPCs controlled by TrafficManager
    - Spawning pedestrian NPCs with AI walker controllers
    """

    def setup(self) -> None:
        """Set up the scenario."""
        # Get spawn point for ego vehicle
        spawn_point = self.get_spawn_point()

        # Spawn ego vehicle using the simulation's spawn method
        # Note: In a real scenario, you'd get the simulation instance
        # For this example, we'll spawn directly using world
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter("vehicle.tesla.model3")[0]
        self.ego_vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)

        print(f"Ego vehicle spawned at {spawn_point.location}")

    def spawn_npcs(self) -> None:
        """Spawn NPCs for the scenario."""
        # Spawn 30 vehicle NPCs
        num_vehicles = self.config.get("num_vehicles", 30)
        self._spawn_vehicle_npcs(num_vehicles)

        # Spawn 20 pedestrian NPCs
        num_pedestrians = self.config.get("num_pedestrians", 20)
        self._spawn_pedestrian_npcs(num_pedestrians)

    def _spawn_vehicle_npcs(self, num_vehicles: int) -> None:
        """Spawn vehicle NPCs using TrafficManager."""
        import numpy as np

        blueprint_library = self.world.get_blueprint_library()
        vehicle_bps = blueprint_library.filter("vehicle.*")

        spawn_points = self.world.get_map().get_spawn_points()
        ego_location = self.ego_vehicle.get_location()

        spawned = 0
        for i in range(len(spawn_points)):
            if spawned >= num_vehicles:
                break

            spawn_point = spawn_points[i]

            # Keep safe distance from ego vehicle
            if spawn_point.location.distance(ego_location) < 10.0:
                continue

            vehicle_bp = np.random.choice(vehicle_bps)

            try:
                vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
                vehicle.set_autopilot(True, self.traffic_manager.get_port())
                self.vehicle_npcs.append(vehicle)
                spawned += 1
            except RuntimeError:
                continue

        print(f"Spawned {len(self.vehicle_npcs)} vehicle NPCs")

    def _spawn_pedestrian_npcs(self, num_pedestrians: int) -> None:
        """Spawn pedestrian NPCs with AI walker controllers."""
        import numpy as np

        blueprint_library = self.world.get_blueprint_library()
        walker_bps = blueprint_library.filter("walker.pedestrian.*")
        controller_bp = blueprint_library.find("controller.ai.walker")

        ego_location = self.ego_vehicle.get_location()
        spawn_radius = 50.0

        # Spawn pedestrians
        for _ in range(num_pedestrians):
            spawn_point = carla.Transform()
            spawn_point.location = ego_location + carla.Location(
                x=np.random.uniform(-spawn_radius, spawn_radius),
                y=np.random.uniform(-spawn_radius, spawn_radius),
                z=0.5,
            )

            walker_bp = np.random.choice(walker_bps)

            try:
                pedestrian = self.world.spawn_actor(walker_bp, spawn_point)
                self.pedestrian_npcs.append(pedestrian)
            except RuntimeError:
                continue

        # Wait for pedestrians to be registered
        self.world.tick()

        # Spawn controllers
        for pedestrian in self.pedestrian_npcs:
            try:
                controller = self.world.spawn_actor(controller_bp, carla.Transform(), pedestrian)
                self.pedestrian_controllers.append(controller)
            except RuntimeError:
                continue

        # Wait for controllers to be registered
        self.world.tick()

        # Start walking behavior
        for controller in self.pedestrian_controllers:
            controller.start()
            # Set random destination
            destination = ego_location + carla.Location(
                x=np.random.uniform(-spawn_radius, spawn_radius),
                y=np.random.uniform(-spawn_radius, spawn_radius),
                z=0.0,
            )
            controller.go_to_location(destination)
            controller.set_max_speed(np.random.uniform(1.0, 2.0))

        print(f"Spawned {len(self.pedestrian_npcs)} pedestrian NPCs")

    def get_spawn_point(self) -> carla.Transform:
        """Get spawn point for ego vehicle."""
        # Use first spawn point by default
        spawn_points = self.world.get_map().get_spawn_points()
        return spawn_points[0] if spawn_points else carla.Transform()

    def run(self) -> None:
        """Run scenario logic each step."""
        # Here you can implement per-step logic such as:
        # - Checking for scenario completion conditions
        # - Logging information
        # - Triggering events
        pass


def main(host: str = "localhost", port: int = 2000, map_name: str = "Town01"):
    """Example of how to run a scenario.

    Args:
        host: CARLA server host address
        port: CARLA server port
        map_name: Map to load (e.g., "Town01", "Town02")
    """
    print(f"Connecting to CARLA server at {host}:{port}")

    # Create simulation
    with CARLASimulation(host=host, port=port, map_name=map_name) as sim:
        # Spawn ego vehicle
        spawn_points = sim.world.get_map().get_spawn_points()
        sim.spawn_ego_vehicle(spawn_points[0])

        # Setup cameras
        sim.setup_cameras()

        # Create scenario
        scenario_config = {
            "num_vehicles": 30,
            "num_pedestrians": 20,
        }
        scenario = SimpleUrbanScenario(
            world=sim.world,
            client=sim.client,
            traffic_manager=sim.traffic_manager,
            config=scenario_config,
            host=host,
            port=port,
        )

        # Run scenario
        sim.run_scenario(scenario, num_steps=1000)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run CARLA simulation with Alpamayo R1",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="CARLA server host address (e.g., 'localhost' or '192.168.1.100')",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=2000,
        help="CARLA server port",
    )
    parser.add_argument(
        "--map",
        type=str,
        default="Town01",
        help="Map to load (e.g., Town01, Town02, Town03, Town04, Town05)",
    )

    args = parser.parse_args()

    main(host=args.host, port=args.port, map_name=args.map)
