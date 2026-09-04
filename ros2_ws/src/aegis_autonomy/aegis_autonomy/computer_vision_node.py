import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
import random

class ComputerVisionNode(Node):
    """
    Cognitive Layer: Computer Vision Models (CNNs).
    Processes high-res optical feeds to identify runways and spot other air traffic.
    """
    def __init__(self):
        super().__init__('computer_vision_node')
        
        # Subscribes to raw camera feed from Hardware Interface
        self.camera_sub = self.create_subscription(Point, '/sensors/vision/eo_ir', self.camera_callback, 10)
        
        # Publishes identified semantic objects (e.g., runways, other planes)
        self.objects_pub = self.create_publisher(Point, '/aegis/perception/identified_objects', 10)
        
        self.get_logger().info('ComputerVisionNode (CNN) initialized.')

    def camera_callback(self, msg):
        """Runs the CNN inference on incoming image frames."""
        # In a full implementation, this runs a YOLOv8 or ResNet forward pass
        # Here we mock the CNN output
        
        if random.random() < 0.1: # 10% chance to spot traffic
            identified_msg = Point()
            identified_msg.x = msg.x # pixel X
            identified_msg.y = msg.y # pixel Y
            identified_msg.z = 2.0   # Object Class: 2.0 = Other Aircraft
            
            self.objects_pub.publish(identified_msg)
            self.get_logger().warn("[CNN VISION] Airborne Traffic detected! Publishing to avoidance planner.")

def main(args=None):
    rclpy.init(args=args)
    node = ComputerVisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
