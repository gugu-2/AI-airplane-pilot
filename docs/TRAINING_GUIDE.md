# Aegis OS — Cognitive RL Training Guide

This document is the definitive guide to training the Aegis OS Autonomous Flight Model using Proximal Policy Optimization (PPO).

---

## 1. What is Reinforcement Learning (RL)?

Unlike traditional AI (like image classifiers or text generators) which require massive, pre-existing datasets collected by humans, **Reinforcement Learning generates its own dataset.** 

### The Student Pilot Analogy
Imagine a student pilot in a flight simulator:
1. They don't know how to fly. They push random buttons (Action).
2. The plane crashes (State).
3. The instructor gives them a failing grade (Penalty/Negative Reward).
4. The student tries again, this time pulling up slightly. The plane flies for 5 seconds before crashing.
5. The instructor says, "Better, but still bad" (Slightly higher Negative Reward).
6. Over millions of attempts, the student figures out exactly what combination of pitch, roll, and thrust keeps the plane in the air.

In RL, the AI is the student, the physics engine is the simulator, and the Reward Function is the instructor. **The AI writes its own training data by experiencing the simulated world.**

---

## 2. The Training Environment

Aegis OS uses a massively parallelized GPU environment (`Vectorized6DOFEnv`) running in PyTorch.

*   **2048 Parallel Worlds:** Instead of training one drone at a time, we spawn 2048 drones in 2048 independent simulated universes simultaneously on the GPU.
*   **6 Degrees of Freedom (6-DOF):** The aircraft can move in X, Y, and Z, and can Pitch, Roll, and Yaw.
*   **The Senses (Inputs):**
    *   **32-Ray LiDAR:** A simulated depth scanner that looks forward in a 120-degree cone to detect obstacles.
    *   **9-DOF Physics State:** The AI knows its current Attitude (pitch, roll, yaw), Velocity (vx, vy, vz), and normalized Distance-to-Target (dx, dy, dz).

---

## 3. The PPO Algorithm

Aegis OS uses **Proximal Policy Optimization (PPO)**. 

### Why PPO over DQN?
Deep Q-Networks (DQN) are great for discrete actions (like pressing 'Up' or 'Down' on a D-pad). Flight requires **continuous control** — you don't just set throttle to 0% or 100%, you need 43.2%. PPO is a continuous control algorithm.

### Actor-Critic Architecture
PPO uses two neural networks working together:
1.  **The Actor:** Looks at the sensors and decides what to do (e.g., "Pitch up 10 degrees, 80% thrust").
2.  **The Critic:** Looks at the sensors and judges how safe the current situation is (e.g., "We are 2 meters from a wall going 50 mph. This is a very bad state. Value = -100").

### GAE (Generalized Advantage Estimation)
GAE is the mathematical trick that helps the AI figure out *which* action caused a crash. If the drone crashes at second 10, was it the decision made at second 9, or the bad turn made at second 2? GAE smoothly assigns credit/blame across time.

---

## 4. Session-by-Session Training Plan

To fit within Google Colab's 5-hour Free T4 limits, training is divided into 5 focused curriculum stages.

### Session 1: Basic 3D Navigation + Obstacle Avoidance
*   **What it learns:** How to take off, stabilize flight, fly from point A to point B, and dodge 8 spherical obstacles.
*   **Rewards:** +300 for reaching goal, -200 for hitting obstacle, -100 for crashing into ground, -0.05 per step (to encourage speed).
*   **Success state:** The drone reaches the target ~80% of the time without crashing.

### Session 2: Vehicle Profiles (Drone vs. Fixed-Wing)
*   **What it learns:** Physics. A drone can hover. A fixed-wing plane will stall and fall out of the sky if it drops below 50 km/h. We inject a `Vehicle Profile Vector` (mass, wingspan, max_thrust, is_vtol) into the network.
*   **Success state:** The same neural network flies a drone slowly through a forest, but flies a plane fast and high.

### Session 3: Landing Scenarios
*   **What it learns:** Precision altitude control and approach vectors. We introduce flat pads, moving ship decks, and forest roads.
*   **Rewards:** Heavy shaping based on descent angle. Approaching a runway at a 3-degree glide slope gives massive points; nose-diving gives penalties.
*   **Success state:** Aircraft successfully bleed off speed and touch down at < 2 m/s vertical velocity.

### Session 4: Terrain Awareness
*   **What it learns:** The world is not flat. We introduce procedural Perlin noise heightmaps (mountains, valleys).
*   **Success state:** The AI learns to fly *over* ridges instead of into them, and can find the flattest patch of ground during a simulated engine failure (crash landing).

### Session 5: Full Curriculum Mix
*   **What it learns:** Generalization. Every episode is a random mix of vehicles, terrain, and weather.
*   **Success state:** A fully robust autopilot capable of handling highly complex, novel situations.

---

## 5. Checkpoint System

Because Google Colab shuts down after 5 hours:
*   **Auto-Save:** The script (`train_rl_checkpointed.py`) saves weights to your mounted Google Drive every 25 updates (roughly every 12 minutes).
*   **Auto-Resume:** When you start Session 2, the script automatically looks in Google Drive, finds the `latest.pth` file from Session 1, and resumes seamlessly.
*   **Graceful Exit:** The script monitors its own uptime. After 4.5 hours, it saves a final checkpoint and gracefully shuts down before Colab can aggressively kill it.

---

## 6. Reading the Training Logs

During training, you will see a scrolling table. Here is what it means:

| Column | Meaning | What to look for |
|--------|---------|------------------|
| **Update** | PPO training cycle (1 to 500) | Should increase steadily. |
| **Global Steps** | Total simulated environment frames | Reaches ~500 Million by the end. |
| **Avg Reward** | The average score across 2048 worlds | Should start at -100 (terrible) and climb to +200 (great). |
| **Goals/Env** | Success rate per drone | Should start near 0.00 and climb to >0.80. |
| **Loss** | Neural network error | Will spike initially, then stabilize. |
| **Elapsed/ETA** | Time tracking | Used to ensure you finish before Colab timeout. |

---

## 7. What the AI Cannot Do (Limitations)

It is critical to understand the boundaries of simulation vs. reality.
> [!WARNING]
> **Not Certified for Real Flight:** This model is not certified by the FAA/EASA.
> **No Weather/Wind:** Currently assumes perfect air density and zero wind shear.
> **No Sensor Noise:** Real LiDAR drops points and real IMUs drift. The simulation currently provides perfect "ground truth" data.
> **Sim-to-Real Gap:** Transferring this to a physical drone will require further Domain Randomization (Stage 5+).
