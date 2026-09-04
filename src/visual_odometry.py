import cv2
import numpy as np

class VisualOdometry:
    """
    Implements Lucas-Kanade Optical Flow to calculate drone velocity (m/s)
    purely from visual camera pixel movement. Crucial for GPS-denied SLAM.
    Fix #9: Added camera_rotation_deg to align camera axes with drone body frame.
    Fix #24: Added camera_matrix/dist_coeffs for lens distortion correction.
    """
    def __init__(self, camera_fov_deg=60.0, frame_width=640, frame_height=480,
                 camera_rotation_deg=0.0, camera_matrix=None, dist_coeffs=None):
        self.prev_gray = None
        self.prev_pts = None
        
        # Fix #9: Rotation angle to align camera-frame velocity to drone-body-frame (NED)
        # 0 = camera faces down, aligned with drone forward=Y. 90 = camera rotated 90deg.
        self.rotation_rad = np.radians(camera_rotation_deg)
        
        # Fix #24: Camera calibration matrices for lens distortion removal
        # If not provided, we skip undistortion (less accurate but still functional)
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        
        # Lucas-Kanade Parameters
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        # Feature finding parameters (Shi-Tomasi corner detection)
        self.feature_params = dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=7,
            blockSize=7
        )
        
        # Camera Intrinsics (approximated from FOV)
        self.fov_rad = np.radians(camera_fov_deg)
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.focal_length_px = (frame_width / 2) / np.tan(self.fov_rad / 2)

    def calculate_velocity(self, current_frame, altitude_m, dt):
        """
        Calculates metric velocity (vx, vy in m/s) based on pixel shift and altitude.
        """
        if current_frame is None:
            return 0.0, 0.0
        
        # Fix #24: Apply lens distortion correction if calibration matrix is provided
        if self.camera_matrix is not None and self.dist_coeffs is not None:
            current_frame = cv2.undistort(current_frame, self.camera_matrix, self.dist_coeffs)
            
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        
        # If this is the first frame, we just find features and return 0
        if self.prev_gray is None or self.prev_pts is None or len(self.prev_pts) < 10:
            self.prev_gray = current_gray
            self.prev_pts = cv2.goodFeaturesToTrack(current_gray, mask=None, **self.feature_params)
            return 0.0, 0.0
            
        # Calculate Optical Flow
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, current_gray, self.prev_pts, None, **self.lk_params)
        
        # Select good tracking points
        if curr_pts is not None and status is not None:
            good_new = curr_pts[status == 1]
            good_old = self.prev_pts[status == 1]
            
            if len(good_new) < 5:
                # Lost tracking, reset features
                self.prev_gray = current_gray
                self.prev_pts = cv2.goodFeaturesToTrack(current_gray, mask=None, **self.feature_params)
                return 0.0, 0.0
                
            # Calculate average pixel shift
            diffs = good_new - good_old
            avg_dx_px = np.mean(diffs[:, 0])
            avg_dy_px = np.mean(diffs[:, 1])
            
            # Convert pixel shift to metric velocity (m/s)
            # Geometry: (pixel_shift / focal_length) * altitude = ground_shift_meters
            vx_meters = (avg_dx_px / self.focal_length_px) * altitude_m
            vy_meters = (avg_dy_px / self.focal_length_px) * altitude_m
            
            velocity_x = vx_meters / dt if dt > 0 else 0
            velocity_y = vy_meters / dt if dt > 0 else 0
            
            # Fix #9: Apply 2D rotation to align camera-frame velocity to drone-body-frame
            # This compensates for any camera mounting angle
            cos_r = np.cos(self.rotation_rad)
            sin_r = np.sin(self.rotation_rad)
            
            # Camera raw axes -> rotated drone body axes
            body_vx = cos_r * velocity_x - sin_r * velocity_y
            body_vy = sin_r * velocity_x + cos_r * velocity_y
            
            # Prepare for next frame
            self.prev_gray = current_gray.copy()
            self.prev_pts = good_new.reshape(-1, 1, 2)
            
            return body_vy, body_vx 
            
        else:
            # Tracking failed completely
            self.prev_gray = current_gray
            self.prev_pts = cv2.goodFeaturesToTrack(current_gray, mask=None, **self.feature_params)
            return 0.0, 0.0
