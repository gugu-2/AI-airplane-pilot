import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, Vector3
import math

class SensorFusionNode(Node):
    def __init__(self):
        super().__init__('sensor_fusion_node')
        
        # Subscriptions to raw hardware (Simulated for now)
        self.gps_sub = self.create_subscription(Point, '/sensors/gps/raw', self.gps_callback, 10)
        self.imu_sub = self.create_subscription(Vector3, '/sensors/imu/raw', self.imu_callback, 10)
        
        # Publisher for the clean, fused state estimation
        self.state_pub = self.create_publisher(Point, '/aegis/state/fused_position', 10)
        
        # EKF State (Simplified for 1D example)
        self.estimated_pos = 0.0
        self.estimated_vel = 0.0
        
        self.get_logger().info('SensorFusionNode initialized. EKF Active.')

    def imu_callback(self, msg):
        """High-frequency IMU updates (Predict Step)"""
        # Simplify: Integrate acceleration to get velocity and position
        dt = 0.01 # 100Hz
        accel = msg.x
        
        self.estimated_pos += self.estimated_vel * dt + 0.5 * accel * (dt ** 2)
        self.estimated_vel += accel * dt

    def gps_callback(self, msg):
        """Low-frequency GPS updates (Update Step)"""
        # Simplify: Basic complimentary filter for demonstration
        # In production, this runs the complex Matrix Math from kalman_filter.py
        raw_gps_x = msg.x
        
        # Trust GPS 20%, IMU Prediction 80% (EKF dynamically calculates this gain)
        kalman_gain = 0.2 
        self.estimated_pos = self.estimated_pos + kalman_gain * (raw_gps_x - self.estimated_pos)
        
        # Publish the clean fused state
        fused_msg = Point()
        fused_msg.x = self.estimated_pos
        fused_msg.y = msg.y # Assuming perfect for now
        fused_msg.z = msg.z
        
        self.state_pub.publish(fused_msg)
        self.get_logger().info(f'Published Fused State: X={self.estimated_pos:.2f}')

def main(args=None):
    rclpy.init(args=args)
    node = SensorFusionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
