import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Point
import random

class VisualNavigationNode(Node):
    """
    Perception & Navigation Layer (Option two.txt)
    Combines Optical Flow (for drift estimation) and Visual Odometry (SLAM) 
    to navigate in GPS-denied environments.
    """
    def __init__(self):
        super().__init__('visual_navigation_node')
        
        # In a real system, subscribes to downward facing camera
        # self.camera_sub = self.create_subscription(...)
        
        # Publishes estimated velocity based on ground texture movement (Optical Flow)
        self.optical_flow_pub = self.create_publisher(Twist, '/sensors/vision/optical_flow', 10)
        
        # Publishes Visual Odometry (SLAM position)
        self.visual_odom_pub = self.create_publisher(Point, '/sensors/nav/visual_odometry', 10)
        
        self.timer = self.create_timer(0.2, self.process_vision_algorithms)
        self.get_logger().info('VisualNavigationNode initialized: Optical Flow & SLAM active.')

    def process_vision_algorithms(self):
        # 1. Optical Flow (Pixel tracking to estimate X/Y drift)
        flow_msg = Twist()
        # Assume slight wind drift that the camera detects by watching the grass move
        flow_msg.linear.x = random.uniform(-0.1, 0.1) # m/s drift
        flow_msg.linear.y = random.uniform(-0.1, 0.1)
        self.optical_flow_pub.publish(flow_msg)
        
        # 2. Visual Odometry / SLAM (Depth estimation mapped to 3D space)
        odom_msg = Point()
        odom_msg.x = 10.0 + random.uniform(-0.5, 0.5)
        odom_msg.y = 15.0 + random.uniform(-0.5, 0.5)
        odom_msg.z = 5.0  + random.uniform(-0.1, 0.1) # Using depth estimation for altitude
        self.visual_odom_pub.publish(odom_msg)
        
        if random.random() < 0.1:
            self.get_logger().info(f"[OPTICAL FLOW] Ground tracking active. Drift X: {flow_msg.linear.x:.2f} m/s")
            self.get_logger().info(f"[VISUAL SLAM] 3D Depth Map updated. Odometry locked.")

def main(args=None):
    rclpy.init(args=args)
    node = VisualNavigationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
