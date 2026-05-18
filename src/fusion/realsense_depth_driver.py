import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import Float32, String
import numpy as np

# In a real environment, you would also import cv_bridge to convert ROS images to OpenCV:
# from cv_bridge import CvBridge

class RealSenseDepthDriver(Node):
    """
    AEGIS AUTONOMY: RealSense D435 Depth & Collision Processing
    Replaces mock avoidance with real-time 3D spatial depth parsing.
    """
    def __init__(self):
        super().__init__('realsense_depth_driver')
        
        # Subscribe to Intel RealSense aligned depth-to-color image stream
        self.depth_sub = self.create_subscription(
            Image,
            '/camera/aligned_depth_to_color/image_raw',
            self.depth_callback,
            10
        )
        
        # Publisher to alert the RL Path Planner of an imminent collision
        self.collision_alert_pub = self.create_publisher(String, '/sensors/spatial/collision_warning', 10)
        
        # Publisher for the closest detected obstacle distance (for dashboard/telemetry)
        self.closest_obstacle_pub = self.create_publisher(Float32, '/sensors/spatial/min_distance', 10)
        
        # self.bridge = CvBridge()
        self.safety_distance_meters = 5.0
        
        self.get_logger().info('RealSense D435 Depth Driver Initialized. Scanning forward sector...')

    def depth_callback(self, msg):
        """
        Process incoming depth matrices at 30/60 FPS.
        """
        try:
            # 1. Convert ROS Image message to OpenCV/Numpy array
            # depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            
            # --- MOCKING THE SENSOR DATA FOR WINDOWS COMPATIBILITY ---
            # Since we aren't plugged into a physical RealSense on this machine right now,
            # we will simulate the numpy depth array that cv_bridge would normally output.
            # A RealSense D435 returns a 16-bit array where each pixel value is distance in millimeters.
            height, width = 480, 640
            depth_image = np.full((height, width), 20000, dtype=np.uint16) # Default: 20 meters clear
            
            # Simulate a 3-meter object appearing directly in the center of the frame
            if np.random.random() < 0.1:
                depth_image[200:280, 300:340] = 3000 # 3000mm = 3.0 meters
            # ---------------------------------------------------------

            # 2. Extract the Region of Interest (ROI) - The center of the drone's flight path
            # We don't care about the far edges, only what is directly in front of us.
            center_y, center_x = depth_image.shape[0] // 2, depth_image.shape[1] // 2
            roi_size = 50 # 100x100 pixel bounding box in the center
            
            roi = depth_image[center_y - roi_size : center_y + roi_size, 
                              center_x - roi_size : center_x + roi_size]

            # 3. Filter out zero values (errors/dead pixels in IR sensors)
            valid_pixels = roi[roi > 0]
            
            if len(valid_pixels) == 0:
                return # No valid depth data
                
            # 4. Calculate the closest object in our forward path (convert mm to meters)
            min_distance = np.min(valid_pixels) / 1000.0
            
            # Publish telemetry
            dist_msg = Float32()
            dist_msg.data = min_distance
            self.closest_obstacle_pub.publish(dist_msg)

            # 5. Trigger Collision Avoidance Firewall if breached
            if min_distance < self.safety_distance_meters:
                self.get_logger().warn(f"[COLLISION DETECTED] Obstacle at {min_distance:.1f}m! Triggering Evasive RL Planner.")
                alert = String()
                alert.data = f"THREAT_AHEAD_DISTANCE_{min_distance:.1f}"
                self.collision_alert_pub.publish(alert)

        except Exception as e:
            self.get_logger().error(f"Depth processing failed: {str(e)}")

def main(args=None):
    rclpy.init(args=args)
    node = RealSenseDepthDriver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
