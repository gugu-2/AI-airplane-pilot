import cv2
import time
from perception import PerceptionModule

class HardwarePerception(PerceptionModule):
    def __init__(self, camera_index=0):
        """
        Initializes connection to a physical camera and the neural network.
        For Jetson CSI cameras, the pipeline string would replace camera_index.
        """
        super().__init__() # Load YOLO model
        print(f"[HardwarePerception] Initializing camera {camera_index}...")
        self.cap = cv2.VideoCapture(camera_index)
        
        # Set resolution for performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

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

    def release(self):
        """
        Releases the camera hardware properly on shutdown.
        """
        if self.camera_active:
            self.cap.release()
            print("[HardwarePerception] Camera released.")
