"""Example scenario demonstrating Alpamayo R1 model-based autonomous driving."""

import argparse

import carla

from .inference import CARLASimulation
from .scenario import BaseScenario


class AutonomousDrivingScenario(BaseScenario):
    """Autonomous driving scenario using Alpamayo R1 model for control.

    This scenario demonstrates:
    - Ego vehicle controlled by Alpamayo R1 model
    - Vehicle NPCs controlled by TrafficManager
    - Pedestrian NPCs with AI walker controllers
    - Optional Rerun visualization of predictions and trajectories
    """

    def setup(self) -> None:
        """Set up the scenario."""
        # Spawn ego vehicle (will automatically handle spawn point allocation and retries)
        self.spawn_ego_vehicle(spawn_point=None, vehicle_model="vehicle.tesla.model3")

    def spawn_npcs(self) -> None:
        """Spawn NPCs for the scenario."""
        num_vehicles = self.config.get("num_vehicles", 30)
        num_pedestrians = self.config.get("num_pedestrians", 20)

        print(f"Spawning {num_vehicles} vehicles and {num_pedestrians} pedestrians...")

        # Use base class methods for spawning
        self.spawn_vehicle_npcs(num_vehicles)
        self.spawn_pedestrian_npcs(num_pedestrians, spawn_radius=60.0)

    def get_spawn_point(self) -> carla.Transform:
        """Get spawn point for ego vehicle."""
        spawn_points = self.world.get_map().get_spawn_points()
        spawn_idx = self.config.get("spawn_point_index", 0)
        return spawn_points[spawn_idx] if spawn_points else carla.Transform()

    def run(self) -> None:
        """Run scenario logic each step."""
        # Scenario logic handled by Alpamayo controller
        # Additional scenario-specific logic can be added here
        pass


def main():
    """Main function to run autonomous driving scenario."""
    parser = argparse.ArgumentParser(
        description="Run Alpamayo R1 autonomous driving in CARLA",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="CARLA server host address",
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
        help="Map to load",
    )
    parser.add_argument(
        "--num-vehicles",
        type=int,
        default=50,
        help="Number of vehicle NPCs",
    )
    parser.add_argument(
        "--num-pedestrians",
        type=int,
        default=30,
        help="Number of pedestrian NPCs",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=2000,
        help="Number of simulation steps",
    )
    parser.add_argument(
        "--spawn-point",
        type=int,
        default=0,
        help="Spawn point index for ego vehicle",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to Alpamayo R1 model checkpoint",
    )
    parser.add_argument(
        "--use-rerun",
        action="store_true",
        help="Enable Rerun visualization",
    )
    parser.add_argument(
        "--target-speed",
        type=float,
        default=5.0,
        help="Target speed in m/s (default: 5.0 m/s = 18 km/h)",
    )

    args = parser.parse_args()

    print(f"Connecting to CARLA server at {args.host}:{args.port}")
    print(f"Configuration:")
    print(f"  Map: {args.map}")
    print(f"  Vehicle NPCs: {args.num_vehicles}")
    print(f"  Pedestrian NPCs: {args.num_pedestrians}")
    print(f"  Simulation steps: {args.num_steps}")
    print(f"  Target speed: {args.target_speed} m/s ({args.target_speed * 3.6:.1f} km/h)")
    print(f"  Rerun visualization: {args.use_rerun}")

    # Load model if path provided
    model = None
    processor = None
    if args.model_path:
        print(f"Loading model from {args.model_path}")
        # TODO: Implement model loading
        # from transformers import AutoModelForCausalLM, AutoProcessor
        # model = AutoModelForCausalLM.from_pretrained(args.model_path)
        # processor = AutoProcessor.from_pretrained(args.model_path)
        # model.eval()
        print("Model loading not yet implemented - using rule-based controller")

    # Create simulation
    with CARLASimulation(host=args.host, port=args.port, map_name=args.map) as sim:
        # Spawn ego vehicle
        spawn_points = sim.world.get_map().get_spawn_points()
        if args.spawn_point >= len(spawn_points):
            print(
                f"Warning: spawn point {args.spawn_point} out of range, using 0 instead"
            )
            args.spawn_point = 0

        sim.spawn_ego_vehicle(spawn_points[args.spawn_point])

        # Setup cameras
        sim.setup_cameras()

        # Create scenario with Alpamayo control enabled
        scenario_config = {
            "num_vehicles": args.num_vehicles,
            "num_pedestrians": args.num_pedestrians,
            "spawn_point_index": args.spawn_point,
        }

        scenario = AutonomousDrivingScenario(
            world=sim.world,
            client=sim.client,
            traffic_manager=sim.traffic_manager,
            config=scenario_config,
            host=args.host,
            port=args.port,
            model=model,
            processor=processor,
            use_alpamayo_control=True,  # Enable Alpamayo controller
            use_rerun=args.use_rerun,
        )

        # Set target speed if controller is initialized
        if scenario.alpamayo_controller is not None:
            scenario.alpamayo_controller.set_target_speed(args.target_speed)

        # Run scenario
        print(f"Running autonomous driving scenario for {args.num_steps} steps...")
        sim.run_scenario(scenario, num_steps=args.num_steps)

        print("Simulation complete!")


if __name__ == "__main__":
    main()
