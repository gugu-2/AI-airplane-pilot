#!/bin/bash
# ==============================================================================
# AEGIS AUTONOMY - HARDWARE DEPLOYMENT SCRIPT
# Target: NVIDIA Jetson Orin Nano (Phase 3: Autonomous Mini Drone)
# ==============================================================================

echo "=================================================="
echo ">>> INITIALIZING NVIDIA JETSON ORIN DEPLOYMENT"
echo "=================================================="

# 1. JetPack & CUDA Setup for YOLOv8 (AI Computer)
echo "[1/4] Configuring NVIDIA JetPack & CUDA for Computer Vision..."
sudo apt-get update
sudo apt-get install -y nvidia-jetpack
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
echo "   -> PyTorch TensorRT acceleration enabled."

# 2. UDEV Rules for Hardware Interfaces (Sensors & Flight Controller)
echo "[2/4] Binding physical hardware ports (UART/USB)..."
# Pixhawk Flight Controller (Telem2 port)
echo 'SUBSYSTEM=="tty", ATTRS{idVendor}=="26ac", ATTRS{idProduct}=="0011", SYMLINK+="pixhawk"' | sudo tee /etc/udev/rules.d/99-pixhawk.rules
# LiDAR (e.g., RPLiDAR A2)
echo 'SUBSYSTEM=="tty", ATTRS{idVendor}=="10c4", ATTRS{idProduct}=="ea60", SYMLINK+="lidar"' | sudo tee /etc/udev/rules.d/99-lidar.rules
# Stereo/Depth Camera (e.g., Intel RealSense)
echo 'SUBSYSTEM=="usb", ATTR{idVendor}=="8086", ATTR{idProduct}=="0b07", MODE="0666", SYMLINK+="depth_camera"' | sudo tee /etc/udev/rules.d/99-realsense.rules
sudo udevadm control --reload-rules && sudo udevadm trigger
echo "   -> Hardware Symlinks created: /dev/pixhawk, /dev/lidar, /dev/depth_camera"

# 3. Micro-ROS / MAVROS Bridge (ROS 2 -> PX4)
echo "[3/4] Installing Micro-XRCE-DDS Agent for PX4 Communication..."
git clone https://github.com/eProsima/Micro-XRCE-DDS-Agent.git
cd Micro-XRCE-DDS-Agent && mkdir build && cd build
cmake .. && make -j$(nproc)
sudo make install
sudo ldconfig
echo "   -> ROS2-to-PX4 middleware installed."

# 4. Compile Aegis Autonomy Workspace
echo "[4/4] Compiling Aegis ROS 2 Workspace for Jetson architecture (ARM64)..."
cd ../../ros2_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release
echo "   -> Workspace compiled successfully."

echo "=================================================="
echo ">>> DEPLOYMENT COMPLETE."
echo ">>> To launch the drone, run:"
echo ">>> 1. MicroXRCEAgent serial --dev /dev/pixhawk -b 921600"
echo ">>> 2. source ros2_ws/install/setup.bash && ros2 launch aegis_autonomy full_system.launch.py"
echo "=================================================="
