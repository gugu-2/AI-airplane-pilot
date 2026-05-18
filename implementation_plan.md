# Autonomous AI Pilot - Development Plan

Based on the documentation provided (`Anti gravity.txt`, `Option one.txt`, `Option two.txt`), the overarching goal is to build an autonomous flight AI, specifically starting with drones to manage complexity and risk before scaling up. The strategy emphasizes a layered architecture (Perception, Decision/Cognition, Control) and a phased roadmap starting entirely in simulation.

This plan outlines how we will iteratively build this system together.

## User Review Required

> [!IMPORTANT]
> The roadmap is extensive. I recommend we focus entirely on **Phase 1 and Phase 2 (Simulation-Only)** for our immediate work. Please review the proposed architecture and tech stack to ensure it aligns with your vision.

## Open Questions

1. **Simulator Choice:** The documents mention Gazebo, Microsoft AirSim, and NVIDIA Isaac Sim. Do you have a preference for which simulator we should set up first? (Gazebo is standard with ROS2, but AirSim/Isaac Sim offer higher fidelity).
2. **Hardware Specs:** Are you working on a machine with a dedicated GPU (e.g., NVIDIA) capable of running local simulations and AI inference?

---

## Proposed Collaborative Roadmap

Building an autonomous flight system is highly complex. We will break this down into manageable, modular phases. We will not touch physical hardware until the simulation is proven robust.

### Phase 1: Environment & Core Stack Setup (Current Focus)

Our first objective is to set up the foundation.

1. **Development Environment:**
   - Set up Python, C++, and necessary build tools.
   - Install **ROS 2** (Robot Operating System) as the middleware.
2. **Flight Stack & Simulation:**
   - Set up **PX4 Autopilot** (Software-In-The-Loop / SITL mode).
   - Integrate a simulator (e.g., Gazebo or AirSim) with PX4.
3. **Communication Bridge:**
   - Establish communication between our custom Python scripts and the simulated drone using **MAVLink** / **MAVSDK** / **ROS 2**.

### Phase 2: Simulation-Only AI Pilot

Once the environment is running, we will build the "Brain" layer in software.

1. **Basic Control:** Write scripts to autonomously command the simulated drone to take off, fly to specific GPS waypoints, and land safely.
2. **Navigation Module:** Implement basic path planning algorithms (e.g., A* or simple waypoint interpolation).
3. **Perception Module (Simulated):** Connect simulated camera/LiDAR feeds to Python. Implement basic computer vision tasks (like recognizing a landing zone using OpenCV/YOLO).
4. **Obstacle Avoidance:** Implement logic to detect obstacles in the simulated environment and alter the flight path dynamically.

### Phase 3: Hardware Translation (Future)

Once Phase 2 is rock-solid, we will adapt the code for physical hardware.

1. **Compute Setup:** Configure an edge AI board (like NVIDIA Jetson) running Linux/ROS2.
2. **Integration:** Connect the Jetson to a physical flight controller (Pixhawk) via serial/MAVLink.
3. **Sensor Calibration:** Calibrate physical cameras, GPS, and IMU.
4. **Tethered/Safe Testing:** Perform controlled real-world tests of the software built in Phase 2.

---

## System Architecture Strategy

We will build the software using a modular, decoupled approach:

1. **Flight Controller Layer (PX4):** Handles low-level stabilization, motor control, and raw sensor input. We will *interact* with this, not rewrite it.
2. **Middleware (ROS 2 / MAVSDK):** Handles message passing between the AI and the drone.
3. **AI Brain Layer (Python/PyTorch):** Where our custom logic lives.
    - `perception_node`: Processes camera/LiDAR data.
    - `navigation_node`: Handles path planning and SLAM.
    - `decision_node`: The state machine handling mission logic and emergency fallbacks.

## Next Actionable Step

If you approve this general approach, our immediate next step will be to **initialize the project repository** and **create a setup script** to install dependencies like ROS2 and PX4 SITL on your machine. 

Let me know if you are ready to begin Phase 1!
