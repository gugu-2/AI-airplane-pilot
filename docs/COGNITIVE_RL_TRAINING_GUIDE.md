# Aegis OS — Deep Reinforcement Learning Training & Hardware Simulation Guide
This guide details the procedures, architectures, and commands required to train the Aegis Cognitive Path Planning Deep Q-Network (DQN) in both high-speed 2D sandboxes and photorealistic 3D simulators. It also details optimization configurations for NVIDIA CUDA, AMD ROCm, Apple Silicon MPS, and multi-core CPU hardware.

---

## 1. DRL Architecture Overview

The cognitive controller is modeled as a Markov Decision Process (MDP) parameterized by:
- **State Space ($S$)**: 5-dimensional normalized vector representing target direction, heading offset, and nearest obstacle distance.
- **Action Space ($A$)**: 5 discrete spatial maneuvers (Maintain Heading, Veer Left, Veer Right, Climb, Descend).
- **Reward Function ($R$)**: Penalizes distance to target and proximity to obstacles, with a massive positive terminal reward for arrival and a massive negative terminal penalty for collisions or geofence breaches.

```
       +---------------------------------------------+
       |                  Environment                |
       |  (Fast 2D Sandbox or Photorealistic 3D)    |
       +---------------------------------------------+
              | State (s)              ^ Action (a)
              v                        |
       +---------------------------------------------+
       |               Cognitive Agent               |
       |  Deep Q-Network (DQN) <--- Replay Buffer    |
       +---------------------------------------------+
```

---

## 2. High-Speed 2D Sandbox Training

Training in a 2D kinematic sandbox (running inside `src/train_rl.py`) allows the model to process up to **10,000 steps per second**, quickly exploring millions of actions and converging on coarse navigation policies before fine-tuning in high-fidelity 3D environments.

### Execution Command
To start 2D training, run the dedicated script from the workspace root:
```bash
python src/train_rl.py
```

### Sandbox Configuration Parameters
You can adjust the parameters within `src/train_rl.py` to refine convergence:
- `capacity=50000`: Experience Replay buffer capacity.
- `BATCH_SIZE = 64`: Mini-batch size sampled during descent steps.
- `gamma = 0.99`: Discount factor for future expected Q-values.
- `eps_decay = 0.995`: Epsilon-greedy exploration decay rate.

---

## 3. High-Fidelity 3D Simulation Training

Once the baseline path-planning model has stabilized in the 2D sandbox, it is deployed to **3D Software-in-the-Loop (SITL)** environments to train on realistic aerodynamics, wind resistance, control latency, and visual feeds.

```
+---------------------------------------------------------------------------------+
|  3D SITL Training Loop                                                         |
|                                                                                 |
|  +-------------------+      MAVLink Commands      +--------------------------+  |
|  | Aegis Brain       | -------------------------> | PX4 SITL / Flight Stack  |  |
|  | (DQN Planner)     | <------------------------- |                          |  |
|  +-------------------+      Telemetry States      +--------------------------+  |
|          ^                                                     |                |
|          |                                                     v                |
|  +-------------------+                                +----------------------+  |
|  | PyTorch / GPU     | <----------------------------- | Gazebo / Isaac Sim   |  |
|  | Optimization      |         Synthetic Frames       | (Rendering/Physics)  |  |
|  +-------------------+                                +----------------------+  |
+---------------------------------------------------------------------------------+
```

### 1. PX4 SITL + Gazebo Simulation
Aegis interfaces directly with PX4 via MAVSDK over UDP ports to collect telemetry and pilot the virtual aircraft inside the Gazebo physics environment.

#### Initialization Sequence
1. In a dedicated Terminal, spin up PX4 SITL inside your simulation workspace:
   ```bash
   cd PX4-Autopilot
   make px4_sitl gazebo-classic_typhoon_h480
   ```
2. In a second Terminal, run the Aegis Pilot in simulation mode:
   ```bash
   python src/main_pilot.py --disable_ws
   ```
   *Note: Under SITL, `main_pilot.py` will connect to localhost UDP port `14540` and execute the DQN waypoints inside Gazebo.*

### 2. NVIDIA Isaac Sim / AirSim Integration
For photorealistic, GPU-accelerated depth camera simulation (Realsense / YOLO perception training), integrate Aegis with NVIDIA Isaac Sim:
1. Start Isaac Sim and load your target synthetic landscape (e.g., custom airport or urban canyons).
2. Attach the **OmniGraph** extension to route camera outputs to a virtual video capture interface (`/dev/video0`).
3. Run the hardware-oriented pilot to process synthetic vision frames:
   ```bash
   python src/main_pilot.py --hardware --jam_gps
   ```

---

## 4. NVIDIA CUDA & GPU Acceleration

Deep Reinforcement Learning performs millions of neural backpropagations. Running on NVIDIA hardware dramatically reduces training time by offloading tensor matrices to CUDA cores.

```
+-----------------------------------------------------------------------+
|  NVIDIA Hardware Stack                                                |
|  +-----------------------------------------------------------------+  |
|  | PyTorch Runtime                                                 |  |
|  | +-------------------------------------------------------------+ |  |
|  | | CUDA / cuDNN (Accelerated Matrix Backpropagation)           | |  |
|  | +-------------------------------------------------------------+ |  |
|  +-----------------------------------------------------------------+  |
|  | NVIDIA Driver Core (Jetson JetPack, L4T, or Desktop Driver)     |  |
|  +-----------------------------------------------------------------+  |
+-----------------------------------------------------------------------+
```

### Activating CUDA in PyTorch
The `rl_models.py` class automatically detects and assigns the optimal compute device. Ensure you have the CUDA runtime enabled:

```python
import torch

# Print device allocation status
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Active Training Hardware: {device}")

# If using GPU, PyTorch will return: 'Active Training Hardware: cuda'
```

### CUDA Memory & Optimization Flags
To maximize GPU throughput on Jetson boards or desktop GPUs, append the following environmental optimizations to your shell profile:

```bash
# Enable the cuDNN auto-tuner to find the fastest convolution algorithms
export TORCH_CUDNN_BENCHMARK=True

# Limit fragmentation on Jetson shared-memory (Unified RAM) architectures
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
```

---

## 5. Alternative Hardware Architectures

If NVIDIA hardware is unavailable, the Aegis training suite adapts automatically to secondary computing engines.

### 1. Apple Silicon (MPS Acceleration)
Apple Silicon chips (M1/M2/M3) utilize **Metal Performance Shaders (MPS)** to accelerate tensors on the integrated GPU:
```python
# MPS configuration in PyTorch
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
```

### 2. AMD Radeon (ROCm Acceleration)
To train on AMD GPUs, use PyTorch built with the ROCm platform.
1. Run the ROCm PyTorch container:
   ```bash
   docker run -it --network=host --device=/dev/kfd --device=/dev/dri rocm/pytorch:latest
   ```
2. Verify GPU recognition:
   ```python
   import torch
   print(torch.cuda.is_available()) # Returns True under ROCm mapping
   ```

### 3. Multi-Core Intel/AMD CPUs (OMP Threads)
If compiling entirely on CPU, bind matrix libraries to use all logical cores in parallel:
```bash
# Set environment variables BEFORE running train_rl.py
export OMP_NUM_THREADS=$(nproc)
export MKL_NUM_THREADS=$(nproc)
```

---

## 6. Model Verification & Telemetry Export

During sandbox or SITL training, the agent regularly flushes loss and reward logs to monitor learning convergence.

```
       Training Loss Curve                      Average Reward per Episode
  Loss                                     Reward
   ^                                        ^
   | \                                      |          /-----\
   |  \                                     |       /--
   |   \---                                 |    /--
   |       \_____                           | /--
   +--------------> Steps                   +--------------> Episodes
```

### Monitoring Convergence
- **Loss Declines**: As the model trains, the loss curve should decrease and settle near zero, indicating stable Q-value predictions.
- **Reward Rises**: The cumulative reward per episode should rise consistently. Episodes will transition from long exploration wanderings to short, direct paths toward targets.

### Model Exporting
Once the model achieves convergence (stable average reward above +9.0 over 100 consecutive episodes), the training script saves the physical network parameters:
```python
# Saves the optimal weights
torch.save(model.state_dict(), 'models/rl_waypoint_planner.pt')
print("[System] Neural weights exported to models/rl_waypoint_planner.pt")
```
This output file `rl_waypoint_planner.pt` is then loaded by `main_pilot.py` to navigate during actual flight missions.
