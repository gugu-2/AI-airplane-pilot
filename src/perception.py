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
            # R10 FIX: Use absolute path so model is found regardless of launch directory
            _models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')
            _root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
            engine_path = os.path.join(_models_dir, 'yolov8n.engine')
            pt_path_models = os.path.join(_models_dir, 'yolov8n.pt')
            # A5 FIX: Also check project root — that's where the file currently lives
            pt_path_root = os.path.join(_root_dir, 'yolov8n.pt')
            
            if os.path.exists(engine_path):
                print(f"[Perception] Loading TensorRT engine: {engine_path}")
                self.model = YOLO(engine_path)
            elif os.path.exists(pt_path_models):
                print(f"[Perception] Loading PyTorch model from models/: {pt_path_models}")
                self.model = YOLO(pt_path_models)
            elif os.path.exists(pt_path_root):
                print(f"[Perception] Loading PyTorch model from project root: {pt_path_root}")
                self.model = YOLO(pt_path_root)
            else:
                print(f"[Perception] No YOLO model found in models/ or project root. Object detection disabled.")
                print(f"[Perception] Download with: python -c \"from ultralytics import YOLO; YOLO('yolov8n.pt')\"")
                self.model = None
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
            largest_contour = max(contours, key=cv2.contourArea)
            
            # R15 FIX: Reject tiny contours (noise, lens flares, small reflections)
            # A real 1m landing pad at 50m altitude subtends ~200px² at 60deg FOV
            if cv2.contourArea(largest_contour) < 500:
                return None
            
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                return (cX, cY)

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
