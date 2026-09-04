# Aegis OS: Master Deployment & Operations Guide

Welcome to the **Aegis Autonomous Flight Operating System**. This document serves as the master technical manual for training the Deep Reinforcement Learning (DRL) agent, configuring real-world hardware, running the 3D physics simulator, and executing flight tests.

---

## Part 1: Training the AI Pilot (2D to 3D Curriculum)

Autonomous flight is extremely complex. Training a neural network from scratch inside a 3D physics simulator (like Gazebo) is computationally prohibitive because modeling aerodynamics takes too much time. Instead, we use **Curriculum Learning**.

### Step 1.1: High-Speed 2D Kinematic Training
We first train the AI in a bare-metal 2D math environment. This allows the NVIDIA GPU to simulate millions of flights per hour without calculating wind physics.
1. Ensure PyTorch is installed with CUDA support.
2. Open your terminal and run:
   ```bash
   python src/train_rl.py
   ```
3. The script will iterate through 5,000 "epochs" (flights). The drone will be rewarded for hitting the target and penalized for crashing.
4. **Result:** Once complete, the trained synaptic weights are automatically saved to `models/aegis_pilot_v1.pth`.

### Step 1.2: 3D Fine-Tuning Integration
When you launch the main flight stack using the `--use_rl` flag, the `RLInferenceEngine` will automatically detect the `.pth` file and load the "smart" weights into memory, allowing the drone to navigate the 3D world using the intuition it built in the 2D simulator.

---

## Part 2: 3D Simulation Testing (PX4 + Gazebo)

To achieve DGCA / FAA certification, the AI must be validated against a true 3D physics engine (Gazebo) and a true flight controller (PX4 SITL).

### Step 2.1: Setting up WSL2
Because PX4 is native to Linux, we run the simulator inside the Windows Subsystem for Linux (WSL2).
1. Open your WSL2 Ubuntu terminal.
2. Run the automated setup script:
   ```bash
   chmod +x scripts/setup_wsl_px4.sh
   ./scripts/setup_wsl_px4.sh
   ```
3. *Note: This will download several gigabytes of data and compile the C++ PX4 codebase.*

### Step 2.2: Launching the Simulator
Inside your WSL2 Ubuntu terminal, launch the physics engine:
```bash
cd ~/PX4-Autopilot
export PX4_SIM_HOST_ADDR=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}')
make px4_sitl gz_x500
```
A 3D Gazebo window will appear showing the quadcopter on a virtual runway.

### Step 2.3: Connecting the Windows AI Brain
Open your standard Windows PowerShell (Command Prompt) and launch the AI:
```powershell
python src/main_pilot.py --connect udp://:14540 --use_rl
```
The AI on Windows will beam MAVLink commands via UDP into the WSL2 physics engine, and you will see the 3D drone take off!

---

## Part 3: Physical Hardware Integration

When moving from the Gazebo Simulator to a physical carbon-fiber drone, Aegis OS runs on an edge companion computer (like an NVIDIA Jetson Orin NX) connected to a Pixhawk Flight Controller.

### Step 3.1: Sensor Hardware (`hardware_perception.py`)
The system expects specific embedded protocols:
*   **Vision:** A physical camera (e.g., Intel RealSense or basic USB Webcam) mounted on the drone for YOLOv8 object detection.
*   **Airspeed:** An MS4525DO Pitot Tube wired to the Jetson's **I2C** pins.
*   **LiDAR:** A TFmini-s or similar laser rangefinder wired to the Jetson's **UART** (Serial) pins.

### Step 3.2: C++ High-Performance Engine
For physical flight, Python is too slow for 1000Hz matrix algebra. You must compile the C++ Extended Kalman Filter (EKF) engine.
*   **On Windows:** Run `scripts\build_cpp.bat` (Requires MinGW/g++).
*   **On Jetson/Linux:** Run `bash scripts/build_cpp.sh`.
*   The Python `state_estimation.py` script will automatically load the resulting `.dll` or `.so` file into memory via `ctypes`, dropping processing latency to microseconds.

### Step 3.3: Connecting to the Physical Flight Controller
Instead of a UDP IP address, you connect the Jetson to the Pixhawk using a physical serial cable (Telemetry 2 port).
```bash
python src/main_pilot.py --connect serial:///dev/ttyTHS1:921600 --hardware --use_rl
```

---

## Part 4: Flight Operations & Testing

### 4.1: The Ground Control Station (GCS)
Aegis features a Palantir-style React Dashboard.
1. Open a new terminal.
2. Navigate to `dashboard/` and run `npm run dev`.
3. Open your browser to `http://localhost:5173`.
4. You will see live Telemetry, the Semantic Memory Map (Radar), live YOLOv8 vision feeds, Aviation Weather, and ADS-B Air Traffic alerts.

### 4.2: Environmental Safety Testing
You can run automated failure simulations to validate the AI's emergency responses:
*   **GPS Jamming:** Run `main_pilot.py --jam_gps`. After 15 seconds, the AI will simulate a military jamming attack, kill the EKF's GPS feed, and rely entirely on Optical Flow (Visual Odometry) Dead Reckoning.
*   **Air Traffic Avoidance:** The `adsb_awareness.py` module generates mock Boeing 737 traffic on the radar. If a blip gets too close, the TCAS system will force the drone to drop altitude.
*   **Weather Overrides:** The `weather_integration.py` engine generates live METAR conditions. If wind speeds exceed 25 knots, a `[WEATHER ALARM]` is triggered in the dashboard.

---
*Aegis Flight OS - Engineered for Autonomous Dominance.*
