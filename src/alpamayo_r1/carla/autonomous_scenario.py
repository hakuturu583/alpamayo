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
        "--no-model",
        action="store_true",
        help="Disable model loading and use rule-based controller only",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Disable video recording to test if it causes segfault",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for spawn point selection (default: 0)",
    )

    args = parser.parse_args()

    print(f"Connecting to CARLA server at {args.host}:{args.port}")
    print(f"Configuration:")
    print(f"  Map: {args.map}")
    print(f"  Vehicle NPCs: {args.num_vehicles}")
    print(f"  Pedestrian NPCs: {args.num_pedestrians}")
    print(f"  Simulation steps: {args.num_steps}")
    print(f"  Random seed: {args.seed}")

    # Load model unless --no-model is specified
    model = None
    processor = None
    if not args.no_model:
        print("Loading Alpamayo R1 model from nvidia/Alpamayo-R1-10B")
        import os
        import torch
        from alpamayo_r1 import helper
        from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1

        # Set CUDA memory management environment variable to reduce fragmentation
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

        # Check CUDA availability
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available. Alpamayo R1 requires GPU.\n"
                "Please ensure CUDA is installed and GPU is accessible."
            )

        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

        # Enable TF32 for better performance and memory efficiency
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        try:
            # Clear CUDA cache before loading model
            torch.cuda.empty_cache()

            print("Loading model...")
            # Use default configuration (same as test_inference.py)
            # Explicit Flash Attention 2 config was causing VRAM issues
            model = AlpamayoR1.from_pretrained(
                "nvidia/Alpamayo-R1-10B", dtype=torch.bfloat16
            ).to("cuda")

            print("Loading processor...")
            processor = helper.get_processor(model.tokenizer)

            print(f"Model loaded successfully!")
            print(f"Model device: {next(model.parameters()).device}")
            print(f"Model dtype: {next(model.parameters()).dtype}")

            # Check model memory usage
            print(f"GPU memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
            print(f"GPU memory reserved: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")

        except Exception as e:
            print(f"\nERROR: Failed to load Alpamayo R1 model")
            print(f"Error message: {e}")
            import traceback
            traceback.print_exc()
            print("\nTo run without model, use --no-model flag")
            raise RuntimeError(f"Model loading failed: {e}") from e
    else:
        print("Model loading disabled (--no-model) - using rule-based controller")

    # Create simulation
    with CARLASimulation(host=args.host, port=args.port, map_name=args.map) as sim:
        # Create scenario with Alpamayo control enabled
        scenario_config = {
            "num_vehicles": args.num_vehicles,
            "num_pedestrians": args.num_pedestrians,
            "spawn_point_index": args.spawn_point,
            "seed": args.seed,
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
        )

        # Store save_video setting for controller initialization
        scenario.save_video = not args.no_video

        # Run scenario
        print(f"Running autonomous driving scenario for {args.num_steps} steps...")
        sim.run_scenario(scenario, num_steps=args.num_steps)

        print("Simulation complete!")


if __name__ == "__main__":
    main()
