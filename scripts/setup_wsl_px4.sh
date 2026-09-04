#!/bin/bash
# ==============================================================================
# PX4 + Gazebo SITL Automated Setup Script for WSL2 (Ubuntu)
# Run this inside your WSL2 terminal!
# ==============================================================================

set -e

echo "Starting PX4 + Gazebo installation for WSL2..."

# 1. Update and install basic dependencies
sudo apt update && sudo apt upgrade -y
sudo apt install -y git wget cmake build-essential python3-pip

# 2. Clone the PX4 Autopilot Repository
if [ ! -d "$HOME/PX4-Autopilot" ]; then
    echo "Cloning PX4-Autopilot..."
    cd $HOME
    git clone https://github.com/PX4/PX4-Autopilot.git --recursive
else
    echo "PX4-Autopilot already exists in $HOME/PX4-Autopilot"
fi

# 3. Run the official ubuntu setup script
echo "Running official PX4 setup script..."
cd $HOME/PX4-Autopilot/Tools/setup
bash ubuntu.sh --no-nuttx --no-sim-tools

# 4. Install Gazebo Harmonic (Recommended for PX4)
echo "Installing Gazebo (Harmonic)..."
sudo curl -sSL https://packages.osrfoundation.org/gazebo.gpg --output /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null
sudo apt-get update
sudo apt-get install -y gz-harmonic

# 5. Install Python dependencies for PX4 MAVSDK
pip3 install kconfiglib jinja2 jsonschema numpy future

# 6. Configure WSL2 Display for Gazebo GUI
echo "Configuring Display variables for WSLg..."
export DISPLAY=:0
export LIBGL_ALWAYS_SOFTWARE=1

echo "====================================================================="
echo "PX4 + Gazebo Setup Complete!"
echo "To launch the simulator, run the following commands in this terminal:"
echo ""
echo "cd ~/PX4-Autopilot"
echo "export PX4_SIM_HOST_ADDR=\$(cat /etc/resolv.conf | grep nameserver | awk '{print \$2}')"
echo "make px4_sitl gz_x500"
echo "====================================================================="
