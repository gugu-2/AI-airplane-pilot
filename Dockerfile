# Aegis OS Dockerfile
# Targets the NVIDIA L4T (Linux for Tegra) base image for Jetson deployment.
# For x86_64 simulation, replace the base image with: python:3.11-slim

FROM nvcr.io/nvidia/l4t-pytorch:r36.2.0-pth2.1-py3

LABEL maintainer="Aegis Flight OS"
LABEL description="Autonomous AI Pilot \u2014 Production Container"

WORKDIR /aegis

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libopencv-dev \
    python3-opencv \
    g++ \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the full project
COPY . .

# Compile the C++ High-Performance EKF Engine (Fix #28: was never compiled in original)
RUN g++ -O3 -shared -fPIC src/cpp_core/fast_ekf.cpp -o src/cpp_core/fast_ekf.so && \
    echo "C++ EKF engine compiled successfully."

# Create the models directory for trained weights
RUN mkdir -p models logs

# Expose WebSocket port for React Dashboard
EXPOSE 8765

# Default entrypoint: runs in simulation mode
# Override with --hardware for physical Pixhawk deployment
ENTRYPOINT ["python", "src/main_pilot.py"]
CMD ["--connect", "udp://:14540"]
