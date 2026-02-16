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
        # Spawn ego vehicle (will automatically handle spawn point allocation and retries)
        self.spawn_ego_vehicle(spawn_point=None, vehicle_model="vehicle.tesla.model3")

    def spawn_npcs(self) -> None:
        """Spawn NPCs for the scenario."""
        num_vehicles = self.config.get("num_vehicles", 30)
        num_pedestrians = self.config.get("num_pedestrians", 20)

        # Use base class methods for spawning
        self.spawn_vehicle_npcs(num_vehicles)
        self.spawn_pedestrian_npcs(num_pedestrians, spawn_radius=50.0)

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
