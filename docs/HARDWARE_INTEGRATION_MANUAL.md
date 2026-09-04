# Aegis Flight OS — Hardware Integration & Calibration Manual
This manual provides the technical specifications, sensor pinouts, camera calibration methods, and Docker deployment commands required to integrate Aegis OS with physical drone hardware (e.g., NVIDIA Jetson boards, Pixhawk autopilots, and I2C/UART sensors).

---

## 1. Pinouts, I2C, and UART Sensor Bus Maps

Aegis connects to avionics sensors via standard embedded buses on the NVIDIA Jetson GPIO header.

```
NVIDIA Jetson GPIO Header (40-Pin)
+--------------------------------------------+
| Pin 1  (3.3V Power)      Pin 2  (5V Power) |
| Pin 3  (I2C_1 SDA) <---> Pitot Tube SDA    |
| Pin 5  (I2C_1 SCL) <---> Pitot Tube SCL    |
| Pin 8  (UART_1 TX) ----> LiDAR RXD         |
| Pin 10 (UART_1 RX) <---- LiDAR TXD         |
| Pin 39 (GND)       <---> Sensor GND        |
+--------------------------------------------+
```

### Sensor Specifications & Settings

| Sensor Type | Interface | Hardware Driver | Default Address / Port | Baud Rate / Settings |
|---|---|---|---|---|
| **Pitot Tube** | I2C | `PitotTubeDriver` | `0x28` on Bus `1` | Standard Mode (100 kHz) |
| **LiDAR Altimeter** | UART | `LidarDriver` | `/dev/ttyS0` | `115200` Baud, 8-N-1 |
| **GPS Module** | UART | `GPSDriver` | `/dev/ttyTHS1` | `9600` Baud, 8-N-1 |
| **IMU (MPU-6050)** | I2C | `IMUDriver` | `0x68` on Bus `1` | Fast Mode (400 kHz) |

---

## 2. Camera Calibration & Visual Odometry Alignment

Visual Odometry (VO) extracts keypoints from successive camera frames to compute velocity in GPS-denied environments. To align 2D camera motion to the 3D drone body frame, precise axes calibration is required.

```
          Drone Body Axes (FRD)             Camera Image Axes (Pixels)
              ^ Forward (x)                       +--------------> u (x_pixel)
              |                                   |
              |                                   |
    Left (y) <+------> Right                      |
              |                                   v v (y_pixel)
              v Down (z)
```

### Camera Calibration Matrix
Camera lens distortion is corrected in real-time using OpenCV:
$$K = \begin{bmatrix} 
f_x & 0 & c_x \\ 
0 & f_y & c_y \\ 
0 & 0 & 1 
\end{bmatrix}, \quad 
D = \begin{bmatrix} k_1 & k_2 & p_1 & p_2 & [k_3] \end{bmatrix}$$
- $f_x, f_y$: Focal lengths (pixels).
- $c_x, c_y$: Principal point (optical center).
- $k_n, p_n$: Radial and tangential distortion coefficients.

### Body-Frame Alignment Rotation
The camera rotation angle $\alpha$ adjusts the visual motion vector to match the physical drone flight path:
$$\begin{bmatrix} v_x \\ v_y \end{bmatrix}_{\text{body}} = 
\begin{bmatrix} 
\cos\alpha & -\sin\alpha \\ 
\sin\alpha & \cos\alpha 
\end{bmatrix} 
\begin{bmatrix} v_x \\ v_y \end{bmatrix}_{\text{camera}}$$
- $\alpha$: Clockwise rotation angle of the camera mount relative to forward heading.

---

## 3. ROS 2 Node Deployment

The Aegis hardware interface and sensor fusion layers run as native ROS 2 nodes, isolating sensor polling from the main flight safety loops.

```bash
# Source the ROS 2 workspace installation
source /opt/ros/humble/setup.bash
source ros2_ws/install/setup.bash
```

### 1. Launching the Hardware Polling Node
The hardware node polls the physical serial/I2C sensors and publishes structured avionics data. We configure custom takeoff coordinates via parameters:
```bash
ros2 run aegis_autonomy hardware_interface_node --ros-args \
  -p home_lat:=37.7749 \
  -p home_lon:=-122.4194
```

### 2. Launching the Sensor Fusion Node
The sensor fusion node runs the Extended Kalman Filter (EKF) over the sensor topics:
```bash
ros2 run aegis_autonomy sensor_fusion_node
```

---

## 4. Docker Deployment on NVIDIA Jetson

For robust deployment on physical edge hardware, the entire Aegis OS stack is containerized, utilizing NVIDIA Jetson CUDA/TensorRT runtimes.

### Container Architecture
```
+-------------------------------------------------------+
|  Aegis OS Docker Container                            |
|  +-------------------------------------------------+  |
|  | Python 3.10 Runtime                             |  |
|  | +------------------+   +----------------------+ |  |
|  | | YOLOv8 TensorRT  |   | PyTorch Inference Engine| |  |
|  | +------------------+   +----------------------+ |  |
|  +-------------------------------------------------+  |
+-------------------------------------------------------+
                           |
        (Mounts I2C & UART /dev nodes directly)
                           v
+-------------------------------------------------------+
|  Host Jetson Linux OS (L4T / NVIDIA JetPack)          |
+-------------------------------------------------------+
```

### Building the Image
Compile the docker image directly on the Jetson edge board:
```bash
docker build -t aegis-os:latest .
```

### Running the Container
Start the container with complete CUDA GPU access, direct I2C bus mounts, and serial ports exposed:
```bash
docker run -d --privileged \
  --runtime nvidia \
  --network host \
  --device /dev/i2c-1:/dev/i2c-1 \
  --device /dev/ttyTHS1:/dev/ttyTHS1 \
  --device /dev/ttyS0:/dev/ttyS0 \
  -v $(pwd)/logs:/aegis/logs \
  aegis-os:latest
```
- `--privileged`: Grants device access for hardware sensor polling.
- `--runtime nvidia`: Enables GPU passthrough for CUDA YOLO inference.
- `--network host`: Maps telemetry WebSockets directly to GCS ports.
