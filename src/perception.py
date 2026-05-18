import cv2
import numpy as np
import os

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[WARNING] ultralytics not installed. YOLOv8 features disabled.")

class PerceptionModule:
    def __init__(self):
        # In a real environment, this would connect to a physical camera or ROS image topic.
        # Here we mock it by generating a synthetic image.
        self.camera_resolution = (480, 640, 3)
        
        # Load YOLOv8 model, preferring optimized TensorRT engine if available
        if YOLO_AVAILABLE:
            engine_path = 'yolov8n.engine'
            pt_path = 'yolov8n.pt'
            
            if os.path.exists(engine_path):
                print(f"[Perception] Loading optimized TensorRT engine: {engine_path}")
                self.model = YOLO(engine_path)
            else:
                print(f"[Perception] TensorRT engine not found. Falling back to PyTorch model: {pt_path}")
                self.model = YOLO(pt_path)
        else:
            self.model = None

    def get_synthetic_image(self):
        """
        Generates a synthetic image simulating what the drone might see.
        """
        # Create a plain blue sky
        img = np.full(self.camera_resolution, (255, 150, 100), dtype=np.uint8)
        
        # Add a green ground
        cv2.rectangle(img, (0, 300), (640, 480), (50, 150, 50), -1)

        # Draw a mock landing pad (red circle)
        cv2.circle(img, (320, 380), 30, (0, 0, 255), -1)

        return img

    def detect_landing_pad(self, image):
        """
        A simple computer vision function to detect a red landing pad in the image.
        """
        # Convert to HSV color space for easier color detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define range of red color in HSV
        lower_red_1 = np.array([0, 120, 70])
        upper_red_1 = np.array([10, 255, 255])
        lower_red_2 = np.array([170, 120, 70])
        upper_red_2 = np.array([180, 255, 255])

        # Threshold the HSV image to get only red colors
        mask1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
        mask2 = cv2.inRange(hsv, lower_red_2, upper_red_2)
        mask = mask1 + mask2

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find the largest contour which should be the landing pad
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Calculate the center of the contour
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                return (cX, cY) # Return center coordinates of the landing pad

        return None # Landing pad not found

    def detect_and_track_objects(self, image):
        """
        Runs YOLOv8 neural network inference with built-in ByteTrack/BoT-SORT tracking.
        Returns a list of dictionaries with class, confidence, bounding box, and unique tracking ID.
        """
        if not self.model:
            return []

        # Run inference with tracking enabled. 'persist=True' keeps track of IDs across frames.
        results = self.model.track(image, persist=True, verbose=False)
        detected_objects = []

        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                cls_name = self.model.names[cls_id]
                conf = float(box.conf[0])
                
                # Retrieve the tracking ID (if the tracker has assigned one)
                track_id = int(box.id[0]) if box.id is not None else -1
                
                # Bounding box coordinates [x1, y1, x2, y2]
                xyxy = box.xyxy[0].tolist()
                
                detected_objects.append({
                    "id": track_id,
                    "class": cls_name,
                    "confidence": conf,
                    "bbox": xyxy
                })
                
        return detected_objects
