# CARLA Simulation Integration

This directory contains code for integrating the Alpamayo R1 model with the CARLA simulator.

## Overview

- **inference.py**: Main CARLA simulation class
- **scenario.py**: Base scenario class with Alpamayo controller support
- **controller.py**: Alpamayo R1 model-based controller for autonomous driving
- **autonomous_scenario.py**: Main autonomous driving scenario (primary entry point)

## Camera Configuration

The system mimics the 7-camera configuration from the NVIDIA PhysicalAI-Autonomous-Vehicles dataset:

| Camera Name | FOV | Position | Purpose |
|------------|-----|----------|---------|
| Front Wide | 120° | Front center | Wide front view |
| Front Tele | 30° | Front center | Telephoto front view |
| Cross Left | 120° | Left side | Left cross-traffic view |
| Cross Right | 120° | Right side | Right cross-traffic view |
| Rear Left | 70° | Rear left | Rear left view |
| Rear Right | 70° | Rear right | Rear right view |
| Rear Tele | 30° | Rear center | Telephoto rear view |

**Resolution**: 1920x1080 (1080p)
**Coordinate System**: Origin at rear axle center, X-axis=forward, Y-axis=left, Z-axis=up

## Installation

```bash
uv sync --group carla
```

This installs:
- `carla>=0.9.15` - CARLA Python API

## Basic Usage

### 1. Starting CARLA Server

#### Local Server

```bash
# In CARLA installation directory
./CarlaUE4.sh
```

#### Remote Server

```bash
# On remote server, start with specific port
./CarlaUE4.sh -carla-rpc-port=2000

# Or use a different port
./CarlaUE4.sh -carla-rpc-port=3000
```

**Note**: For remote connections, ensure the specified port is open in the firewall.

### 2. Running Simulation

```python
from alpamayo_r1.carla import CARLASimulation, BaseScenario

# Local connection
with CARLASimulation(host="localhost", port=2000, map_name="Town01") as sim:
    # ... (same as below)

# Remote connection
with CARLASimulation(host="192.168.1.100", port=2000, map_name="Town01") as sim:
    # Spawn ego vehicle
    spawn_points = sim.world.get_map().get_spawn_points()
    sim.spawn_ego_vehicle(spawn_points[0])

    # Setup cameras
    sim.setup_cameras()

    # Spawn vehicle NPCs (controlled by TrafficManager)
    vehicles = sim.spawn_vehicle_npcs(num_vehicles=30)

    # Spawn pedestrian NPCs (controlled by AI walker)
    pedestrians, controllers = sim.spawn_pedestrian_npcs(num_pedestrians=20)

    # Simulation loop
    for step in range(1000):
        sim.world.tick()

        # Get camera images
        images = sim.get_camera_images()

        # Implement Alpamayo R1 model inference and control here
        # ...
```

## Creating Custom Scenarios

You can create custom scenarios by inheriting from the `BaseScenario` class:

```python
from alpamayo_r1.carla import BaseScenario
import carla

class MyCustomScenario(BaseScenario):
    def setup(self) -> None:
        """Setup the scenario"""
        # Spawn ego vehicle
        spawn_point = self.get_spawn_point()
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter("vehicle.tesla.model3")[0]
        self.ego_vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)

    def spawn_npcs(self) -> None:
        """Spawn NPCs"""
        # Spawn vehicle and pedestrian NPCs
        pass

    def get_spawn_point(self) -> carla.Transform:
        """Return ego vehicle spawn point"""
        spawn_points = self.world.get_map().get_spawn_points()
        return spawn_points[0]

    def run(self) -> None:
        """Logic executed each step"""
        # Scenario-specific processing
        pass

# Run scenario
with CARLASimulation(host="localhost", port=2000) as sim:
    sim.spawn_ego_vehicle()
    sim.setup_cameras()

    scenario = MyCustomScenario(
        world=sim.world,
        client=sim.client,
        traffic_manager=sim.traffic_manager,
        config={"key": "value"},
        host=sim.host,
        port=sim.port,
        model=model,  # Optional: Alpamayo R1 model
        processor=processor,  # Optional: model processor
        use_alpamayo_control=True,  # Enable autonomous control
    )

    sim.run_scenario(scenario, num_steps=1000)
```

## Alpamayo Controller

The `AlpamayoController` class provides model-based autonomous control for the ego vehicle:

```python
from alpamayo_r1.carla import CARLASimulation, AlpamayoController

with CARLASimulation(host="localhost", port=2000) as sim:
    # Spawn ego vehicle
    sim.spawn_ego_vehicle()

    # Setup cameras
    sim.setup_cameras()

    # Create Alpamayo controller
    controller = AlpamayoController(
        ego_vehicle=sim.ego_vehicle,
        cameras=sim.cameras,
        model=model,  # Your loaded Alpamayo R1 model
        processor=processor,
        control_frequency=10.0,  # 10 Hz control updates
    )

    # Simulation loop
    for step in range(1000):
        snapshot = sim.world.tick()
        images = sim.get_camera_images()

        # Update controller (runs inference and applies control)
        controller.update(snapshot, images)
```

### Controller Features

- **Model Inference**: Processes camera images with Alpamayo R1 model
- **Trajectory Prediction**: Extracts predicted trajectories from model outputs
- **Vehicle Control**: Applies steering, throttle, and brake based on predictions
- **State Tracking**: Maintains ego vehicle position and rotation history

### Running Autonomous Scenario

```bash
# With Alpamayo R1 model (default - requires GPU)
python -m alpamayo_r1.carla.autonomous_scenario

# Without model (rule-based controller only)
python -m alpamayo_r1.carla.autonomous_scenario --no-model

# Remote server with custom parameters
python -m alpamayo_r1.carla.autonomous_scenario \
    --host 192.168.1.100 \
    --port 2000 \
    --map Town05 \
    --num-vehicles 100 \
    --num-pedestrians 50 \
    --target-speed 8.0
```

**Note**:
- By default, loads `nvidia/Alpamayo-R1-10B` from HuggingFace Hub
- Model loading uses `AlpamayoR1.from_pretrained()` with `dtype=torch.bfloat16`
- **Requires CUDA-capable GPU** - will fail if CUDA is not available
- Use `--no-model` to run with rule-based controller (no GPU required)

### Autonomous Scenario Arguments

- `--host`: CARLA server host (default: localhost)
- `--port`: CARLA server port (default: 2000)
- `--map`: Map to load (default: Town01)
- `--num-vehicles`: Number of vehicle NPCs (default: 50)
- `--num-pedestrians`: Number of pedestrian NPCs (default: 30)
- `--num-steps`: Simulation steps (default: 2000)
- `--spawn-point`: Ego vehicle spawn point index (default: 0)
- `--no-model`: Disable model loading (use rule-based controller)
- `--target-speed`: Target speed in m/s (default: 5.0)

## NPC Control

### Vehicle NPCs (TrafficManager)

Vehicle NPCs are automatically controlled by TrafficManager and follow traffic rules:

```python
vehicles = sim.spawn_vehicle_npcs(
    num_vehicles=50,      # Number of vehicles
    safe_distance=10.0    # Minimum distance from ego vehicle
)
```

### Pedestrian NPCs (AI Walker)

Pedestrian NPCs are controlled by `controller.ai.walker` and walk randomly:

```python
pedestrians, controllers = sim.spawn_pedestrian_npcs(
    num_pedestrians=30,   # Number of pedestrians
    spawn_radius=50.0     # Spawn radius from ego vehicle
)
```

## Troubleshooting

### Cannot Connect to CARLA Server

- Verify CARLA server is running
- Check host name and port number are correct
- Check firewall settings

### Cannot Get Camera Images

- Verify `sim.setup_cameras()` has been called
- Verify ego vehicle has been spawned
- Verify `world.tick()` is being called

### NPCs Not Spawning

- Verify map has sufficient spawn points
- Verify synchronous mode is enabled
- Check logs for error messages

## References

- [CARLA Documentation](https://carla.readthedocs.io/)
- [NVIDIA PhysicalAI-AV Dataset](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles)
- [PhysicalAI-AV Developer Kit](https://github.com/NVlabs/physical_ai_av)
