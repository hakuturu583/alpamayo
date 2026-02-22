<div align="center">

# 🏔️ Alpamayo 1

### Bridging Reasoning and Action Prediction for Generalizable Autonomous Driving

[![HuggingFace](https://img.shields.io/badge/🤗%20Model-Alpamayo--R1--10B-blue)](https://huggingface.co/nvidia/Alpamayo-R1-10B)
[![arXiv](https://img.shields.io/badge/arXiv-2511.00088-b31b1b.svg)](https://arxiv.org/abs/2511.00088)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](./LICENSE)

</div>

_Note: Following the release of [NVIDIA Alpamayo](https://nvidianews.nvidia.com/news/alpamayo-autonomous-vehicle-development) at CES 2026, Alpamayo-R1 has been renamed to Alpamayo 1._

> **📖 Please read the [HuggingFace Model Card](https://huggingface.co/nvidia/Alpamayo-R1-10B) first!**
> The model card contains comprehensive details on model architecture, inputs/outputs, licensing, and tested hardware configurations. This GitHub README focuses on setup, usage, and frequently asked questions.

## Requirements

| Requirement | Specification |
|-------------|---------------|
| **Python** | 3.12.x (see `pyproject.toml`) |
| **GPU** | NVIDIA GPU with ≥24 GB VRAM (e.g., RTX 3090, RTX 4090, A5000, H100) |
| **OS** | Linux (tested); other platforms unverified |

> ⚠️ **Note**: GPUs with less than 24 GB VRAM will likely encounter CUDA out-of-memory errors.

## Installation

### 1. Install uv (if not already installed)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

### 2. Set up the environment

```bash
uv venv ar1_venv
source ar1_venv/bin/activate
uv sync --active
```

### 3. Authenticate with HuggingFace

The model requires access to gated resources. Request access here:
- 🤗 [Physical AI AV Dataset](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles)
- 🤗 [Alpamayo Model Weights](https://huggingface.co/nvidia/Alpamayo-R1-10B)

Then authenticate using the HuggingFace CLI:

```bash
# Install huggingface-cli if not already installed (included in transformers)
pip install huggingface_hub

# Login with your token
huggingface-cli login
```

Get your access token at: https://huggingface.co/settings/tokens

> 💡 **Tip**: For more details on HuggingFace authentication, see the [official documentation](https://huggingface.co/docs/huggingface_hub/guides/cli).

## Running Inference

### Test script

NOTE: This script will download both some example data (relatively small) and the model weights (22 GB).
The latter can be particularly slow depending on network bandwidth.
For reference, it takes around 2.5 minutes on a 100 MB/s wired connection.

```bash
python src/alpamayo_r1/test_inference.py
```

In case you would like to obtain more trajectories and reasoning traces, please feel free to change
the `num_traj_samples=1` argument to a higher number (Line 60).

### Interactive notebook

We provide a notebook with similar inference code at `notebook/inference.ipynb`.

## Relationship with the Paper

Alpamayo 1 implements the architecture described in our paper [*"Alpamayo-R1: Bridging Reasoning and Action Prediction for Generalizable Autonomous Driving in the Long Tail
"*](https://arxiv.org/abs/2511.00088), including:

| Feature | Paper Description | This Release (v1.0) |
|---------|-------------------|---------------------|
| **Chain-of-Causation (CoC) reasoning** | Hybrid auto-labeling with human in the loop for reasoning traces | ✅ Included |
| **Vision-Language-Action architecture** | Cosmos-Reason backbone + action expert | ✅ Included |
| **Trajectory prediction** | 6.4s horizon, 64 waypoints at 10 Hz | ✅ Included |
| **RL post-training** | Reinforcement learning for reasoning/action consistency | ❌ Not in this release |
| **Route/navigation conditioning** | Explicit navigation or route inputs | ❌ Not in this release |
| **Meta-actions/General VQA** | High-level behavior and visual question answering | ❌ Not in this release |

The current release focuses on the core supervised learning components. RL post-training and route conditioning are potential candidates for future releases. Stay tuned!

## Frequently Asked Questions (FAQ)

<details>
<summary><strong>Does the 10B model accept navigation/route inputs?</strong></summary>

While we have experimented with route conditioning capabilities, the released model does **not** include this feature. The current release takes multi-camera video and egomotion history as inputs, without explicit navigation or route inputs (e.g., waypoints, turn-by-turn navigation instructions).

</details>

<details>
<summary><strong>Does the model produce meta-actions or support general VQA?</strong></summary>

While we have experimented with meta-action and general VQA capabilities, the released model does **not** include these features. Alpamayo 1 is designed specifically for trajectory prediction with Chain-of-Causation reasoning, producing trajectory + reasoning trace outputs.

</details>

<details>
<summary><strong>Was the 10B model post-trained with Reinforcement Learning (RL)?</strong></summary>

No. The current 10B model release has **not** undergone RL post-training. While the paper describes RL stages for improving reasoning quality and action consistency, this release focuses on the supervised learning components. As mentioned above, we may release RL post-trained models in future releases.

</details>

<details>
<summary><strong>What are the minimum GPU requirements?</strong></summary>

You need an NVIDIA GPU with at least **24 GB VRAM** for inference. Tested configurations include RTX 3090, A100, and H100. Running on GPUs with less memory (e.g., 16 GB) will likely result in CUDA out-of-memory errors.

</details>

<details>
<summary><strong>Can I use this model in production / commercial applications?</strong></summary>

No. The model weights are released under a **non-commercial license**. This release is intended for research, experimentation, and evaluation purposes only. See the [License](#license) section and the [HuggingFace Model Card](https://huggingface.co/nvidia/Alpamayo-R1-10B) for details.

</details>

## Project Structure

```
alpamayo/
├── notebook/
│   └── inference.ipynb                  # Example notebook
├── src/
│   └── alpamayo_r1/
│       ├── action_space/
│       │   └── ...                      # Action space definitions
│       ├── diffusion/
│       │   └── ...                      # Diffusion model components
│       ├── geometry/
│       │   └── ...                      # Geometry utilities and modules
│       ├── models/
│       │   ├── ...                      # Model components and utils functions
│       ├── __init__.py                  # Package marker
│       ├── config.py                    # Model and experiment configuration
│       ├── helper.py                    # Utility functions
│       ├── load_physical_aiavdataset.py # Dataset loader
│       ├── test_inference.py            # Inference test script
├── pyproject.toml                       # Project dependencies
└── uv.lock                              # Locked dependency versions
```

## Troubleshooting

## Practical controller recipe: Pure Pursuit + Speed PID

If you are integrating Alpamayo 1 into a lightweight downstream stack, a practical baseline is:

- **Lateral control**: Pure Pursuit tracking of the predicted trajectory.
- **Longitudinal control**: Speed PID tracking a target speed profile extracted from the same trajectory.

> ⚠️ This is still a research-oriented integration pattern and **not** a production-safe AV controller.

### 1) Data flow at 10 Hz (reference)

1. Run Alpamayo inference and get a future trajectory (e.g., 64 waypoints / 6.4 s).
2. Convert waypoints to a local ego frame and compute per-waypoint heading/curvature/speed target.
3. Select a lookahead point for Pure Pursuit.
4. Compute steering from Pure Pursuit.
5. Compute acceleration/brake command from speed PID.
6. Apply actuator bounds and rate limits before sending commands.

### 2) Pure Pursuit lateral control

Given wheelbase `L`, lookahead distance `Ld`, and lookahead point `(x_L, y_L)` in the ego frame:

```text
alpha = atan2(y_L, x_L)
kappa_cmd = 2 * sin(alpha) / Ld
steer_cmd = atan(L * kappa_cmd)
```

Practical tuning:

- Start with a speed-adaptive lookahead:
  - `Ld = clamp(Ld_min + k_v * v, Ld_min, Ld_max)`
- Example initial values for passenger-car scale:
  - `Ld_min = 3.0 m`
  - `Ld_max = 15.0 m`
  - `k_v = 0.6 s`

### 3) Speed PID longitudinal control

Use nearest/future trajectory waypoint speed as `v_ref` and current ego speed as `v`:

```text
e_v = v_ref - v
u_pid = Kp * e_v + Ki * integral(e_v) + Kd * d(e_v)/dt
```

Then map `u_pid` to accel/brake commands with saturation and anti-windup.

Suggested initialization (must be re-tuned per platform):

- `Kp = 0.8`
- `Ki = 0.15`
- `Kd = 0.02`
- integral clamp: `[-1.5, 1.5]`

### 4) Replan hysteresis (to reduce command flicker)

For 10 Hz replanning, avoid abrupt command switching by blending with the previous plan:

```text
u_t = (1 - w_hys) * u_new_t + w_hys * u_prev_shifted_t
```

- `u_t` can be `[accel_t, kappa_t]`.
- `u_prev_shifted_t` is previous cycle command sequence shifted by 1 step.
- Start with `w_hys = 0.2` and dynamically reduce it during urgent maneuvers.

Simple dynamic schedule:

- normal driving: `w_hys = 0.2`
- low TTC or large lateral error: `w_hys = 0.05`

### 5) Minimal pseudocode

```python
# each control tick (10 Hz)
traj = planner.predict(observation)  # Alpamayo output trajectory

# lateral target
Ld = clamp(Ld_min + k_v * ego_speed, Ld_min, Ld_max)
pt = choose_lookahead_point(traj, Ld)
alpha = math.atan2(pt.y, pt.x)
kappa_new = 2.0 * math.sin(alpha) / max(Ld, 1e-3)

# longitudinal target
v_ref = sample_speed_reference(traj)
e_v = v_ref - ego_speed
accel_new = speed_pid.step(e_v, dt)

# hysteresis blend
w_hys = 0.05 if (ttc < 2.0 or abs(lat_err) > 0.4) else 0.2
accel_cmd = (1 - w_hys) * accel_new + w_hys * prev_accel
kappa_cmd = (1 - w_hys) * kappa_new + w_hys * prev_kappa

# bounds + rate limits
accel_cmd = clip(accel_cmd, accel_min, accel_max)
kappa_cmd = clip(kappa_cmd, kappa_min, kappa_max)
accel_cmd = rate_limit(accel_cmd, prev_accel, da_max)
kappa_cmd = rate_limit(kappa_cmd, prev_kappa, dkappa_max)

send_control(accel_cmd, kappa_cmd)
prev_accel, prev_kappa = accel_cmd, kappa_cmd
```

### Flash Attention issues

The model uses Flash Attention 2 by default. If you encounter compatibility issues:

```python
# Use PyTorch's scaled dot-product attention instead
config.attn_implementation = "sdpa"
```

### CUDA out-of-memory errors

If you encounter OOM errors:
1. Ensure you have a GPU with at least 24 GB VRAM
2. Reduce `num_traj_samples` if generating multiple trajectories
3. Close other GPU-intensive applications

## License

- **Inference code**: Apache License 2.0 - see [LICENSE](./LICENSE) for details.
- **Model weights**: Non-commercial license - see [HuggingFace Model Card](https://huggingface.co/nvidia/Alpamayo-R1-10B) for details.

## Disclaimer

Alpamayo 1 is a pre-trained reasoning model designed to accelerate research and development in the autonomous vehicle (AV) domain. It is intended to serve as a foundation for a range of AV-related use cases-from instantiating an end-to-end backbone for autonomous driving to enabling reasoning-based auto-labeling tools. In short, it should be viewed as a building block for developing customized AV applications.

Important notes:

- Alpamayo 1 is provided solely for research, experimentation, and evaluation purposes.
- Alpamayo 1 is not a fully fledged driving stack. Among other limitations, it lacks access to critical real-world sensor inputs, does not incorporate required diverse and redundant safety mechanisms, and has not undergone automotive-grade validation for deployment.

By using this model, you acknowledge that it is a research tool intended to support scientific inquiry, benchmarking, and exploration—not a substitute for a certified AV stack. The developers and contributors disclaim any responsibility or liability for the use of the model or its outputs.

## Citation

If you use Alpamayo 1 in your research, please cite:

```bibtex
@article{nvidia2025alpamayo,
      title={{Alpamayo-R1}: Bridging Reasoning and Action Prediction for Generalizable Autonomous Driving in the Long Tail},
      author={NVIDIA and Yan Wang and Wenjie Luo and Junjie Bai and Yulong Cao and Tong Che and Ke Chen and Yuxiao Chen and Jenna Diamond and Yifan Ding and Wenhao Ding and Liang Feng and Greg Heinrich and Jack Huang and Peter Karkus and Boyi Li and Pinyi Li and Tsung-Yi Lin and Dongran Liu and Ming-Yu Liu and Langechuan Liu and Zhijian Liu and Jason Lu and Yunxiang Mao and Pavlo Molchanov and Lindsey Pavao and Zhenghao Peng and Mike Ranzinger and Ed Schmerling and Shida Shen and Yunfei Shi and Sarah Tariq and Ran Tian and Tilman Wekel and Xinshuo Weng and Tianjun Xiao and Eric Yang and Xiaodong Yang and Yurong You and Xiaohui Zeng and Wenyuan Zhang and Boris Ivanovic and Marco Pavone},
      year={2025},
      journal={arXiv preprint arXiv:2511.00088},
}
```
