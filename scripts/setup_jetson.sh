#!/bin/bash

# setup_jetson.sh
# Automates the installation of required libraries on an NVIDIA Jetson running Ubuntu 20.04/22.04

set -e

echo "========================================"
echo "Starting NVIDIA Jetson AI Brain Setup..."
echo "========================================"

# 1. Update and install basic dependencies
echo "[1/4] Updating system and installing base dependencies..."
sudo apt-get update && sudo apt-get upgrade -y
sudo apt-get install -y python3-pip python3-dev build-essential cmake git pkg-config python3-venv

# 2. Setup Virtual Environment
echo "[2/4] Setting up Python virtual environment..."
cd ~/fly # Assuming the repo is cloned here
python3 -m venv venv
source venv/bin/activate

# 3. Install MAVSDK and OpenCV
echo "[3/4] Installing MAVSDK, OpenCV, and dependencies..."
pip install --upgrade pip
pip install mavsdk numpy opencv-python

# 4. Install PyTorch for Jetson (Jetpack specific)
echo "[4/4] Installing PyTorch for NVIDIA Jetson..."
echo "Note: Standard pip install torch does not work on Jetson ARM64 architecture with CUDA."
echo "Downloading PyTorch wheel optimized for Jetpack 5.1 (Ubuntu 20.04)..."
wget https://developer.download.nvidia.com/compute/redist/jp/v51/pytorch/torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl -O torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl
pip install torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl

echo "========================================"
echo "Setup Complete!"
echo "To activate the environment, run: source ~/fly/venv/bin/activate"
echo "Make sure to connect the Pixhawk to the Jetson UART pins (ttyTHS1)."
echo "========================================"
