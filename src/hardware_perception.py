import cv2
import time
import random
import sys
import os

from perception import PerceptionModule
from visual_odometry import VisualOdometry

# Add embedded driver path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
try:
    from embedded.pitot_tube_driver import PitotTubeDriver
except ImportError:
    PitotTubeDriver = None

class HardwarePerception(PerceptionModule):
    def __init__(self, camera_index=0, camera_rotation_deg=0.0,
                 camera_matrix=None, dist_coeffs=None):
        """
        Initializes connection to a physical camera and the neural network.
        A7 FIX: camera_rotation_deg is now accepted and forwarded to VisualOdometry
                so the camera mounting angle can be configured.
        camera_rotation_deg: Clockwise angle in degrees to align camera axes with drone body frame.
                             0 = camera faces directly down, forward = North.
        """
        super().__init__()
        print(f"[HardwarePerception] Initializing camera {camera_index} (rotation={camera_rotation_deg}deg)...")
        self.cap = cv2.VideoCapture(camera_index)
        
        self.pitot_tube = None
        if PitotTubeDriver:
            try:
                self.pitot_tube = PitotTubeDriver()
                print("[HardwarePerception] Pitot Tube initialized via I2C.")
            except Exception as e:
                print(f"[HardwarePerception] Pitot Tube not detected, using mock data. ({e})")
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # A7 FIX: Pass camera_rotation_deg (and optional calibration) to VisualOdometry
        self.vo = VisualOdometry(
            frame_width=640,
            frame_height=480,
            camera_rotation_deg=camera_rotation_deg,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs
        )
        self.last_vo_time = time.time()

        if not self.cap.isOpened():
            print("[WARNING] Could not open physical camera! Falling back to simulated feed if requested.")
            self.camera_active = False
        else:
            self.camera_active = True
            
    def get_real_image(self):
        """
        Captures a live frame from the hardware camera.
        """
        if not self.camera_active:
            return None
            
        ret, frame = self.cap.read()
        if ret:
            return frame
        else:
            print("[ERROR] Failed to grab frame from physical camera.")
            return None

    def read_embedded_sensors(self, current_frame=None, altitude_m=10.0, vision_objects_count=0):
        """
        Polls the I2C/UART sensors and returns a unified telemetry dictionary.
        Falls back to mock simulation data if the real hardware is missing (e.g. on Windows).
        """
        # Read Airspeed
        airspeed = None
        if self.pitot_tube:
            airspeed = self.pitot_tube.read_airspeed()
            
        if airspeed is None:
            # Mock airspeed around 15 m/s (cruise speed)
            airspeed = 15.0 + random.uniform(-1.0, 1.0)
            
        # Read LiDAR (Mocked for now since UART is disconnected on dev machine)
        lidar_distance = altitude_m if altitude_m > 0 else 100.0 + random.uniform(-2.0, 2.0)
        
        # Calculate Visual Odometry velocity if frame is provided
        vo_vx, vo_vy = 0.0, 0.0
        if current_frame is not None:
            current_time = time.time()
            dt = current_time - self.last_vo_time
            if dt > 0.01: # Cap at 100Hz max
                vo_vx, vo_vy = self.vo.calculate_velocity(current_frame, altitude_m, dt)
                self.last_vo_time = current_time
        
        return {
            "pitot_airspeed": airspeed,
            "lidar_distance": lidar_distance,
            "vision_targets_count": vision_objects_count,
            "vo_velocity_x": vo_vx,
            "vo_velocity_y": vo_vy
        }

    def release(self):
        """
        Releases the camera and embedded hardware properly on shutdown.
        """
        if self.camera_active:
            self.cap.release()
            print("[HardwarePerception] Camera released.")
        if self.pitot_tube:
            self.pitot_tube.close()
            print("[HardwarePerception] Pitot tube I2C bus closed.")
