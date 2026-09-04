# ⚙️ Aegis Autonomy Dependencies

Building autonomous flight systems requires robust, real-time-capable tools. The industry standard is Linux, ROS 2, and flight stacks like PX4.

## Core Flight Stack
- **OS**: Linux (Ubuntu 22.04 recommended) or Windows Subsystem for Linux (WSL2)
- **Flight Controller**: PX4 Autopilot
- **Robotics Middleware**: ROS 2 (Humble)
- **Simulation**: Gazebo (PX4 SITL)

## Python Libraries
Run the following to install the Python dependencies required for the AI modules and the Ground Control Station:

```bash
# For YOLOv8 / RL Neural Networks
pip install torch torchvision torchaudio  

# For PX4 MAVLink communication
pip install mavsdk                        

# For the Ground Control Web Dashboard
pip install flask flask-socketio          

# For RealSense Depth Computer Vision
pip install numpy opencv-python           
```
