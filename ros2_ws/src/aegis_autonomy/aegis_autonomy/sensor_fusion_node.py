import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, Vector3
import math
import sys
import os

# R8 FIX: Import the real EKF from the main src package
# instead of using the fixed-gain complementary filter
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../src')))
try:
    from state_estimation import ExtendedKalmanFilter
    EKF_AVAILABLE = True
except ImportError:
    EKF_AVAILABLE = False
    print("[SensorFusion] WARNING: Could not import ExtendedKalmanFilter. Using fallback filter.")

class SensorFusionNode(Node):
    def __init__(self):
        super().__init__('sensor_fusion_node')
        
        # Subscriptions to raw hardware
        self.gps_sub = self.create_subscription(Point, '/sensors/nav/gps', self.gps_callback, 10)
        self.imu_sub = self.create_subscription(Vector3, '/sensors/nav/imu', self.imu_callback, 10)
        
        # Publisher for the clean, fused state estimation
        self.state_pub = self.create_publisher(Point, '/aegis/state/fused_position', 10)
        
        # 3D State: Position and Velocity (x=lat, y=lon, z=alt)
        self.est_pos = {'x': 0.0, 'y': 0.0, 'z': 0.0}
        self.est_vel = {'x': 0.0, 'y': 0.0, 'z': 0.0}
        self.initialized_gps = False
        
        # R8 FIX: Use the real EKF instead of a fixed 0.2 gain filter
        self.ekf = None  # Initialized on first GPS fix
        
        self.get_logger().info('SensorFusionNode initialized. Real EKF Active.' if EKF_AVAILABLE else 'SensorFusionNode initialized. Fallback filter active.')

    def imu_callback(self, msg):
        """High-frequency IMU updates (Predict Step)"""
        if not self.initialized_gps:
            return # Wait for absolute anchor
            
        dt = 0.01 # 100Hz
        
        # Integrate acceleration
        # In reality, must subtract gravity vector dynamically based on orientation
        accel_x = msg.x
        accel_y = msg.y
        accel_z = msg.z - 9.81 # Remove nominal gravity roughly
        
        # Simple flat-earth integration (1m ~ 1m in lat/lon is inaccurate, but serves the structural purpose)
        self.est_pos['x'] += self.est_vel['x'] * dt + 0.5 * accel_x * (dt ** 2)
        self.est_pos['y'] += self.est_vel['y'] * dt + 0.5 * accel_y * (dt ** 2)
        self.est_pos['z'] += self.est_vel['z'] * dt + 0.5 * accel_z * (dt ** 2)
        
        self.est_vel['x'] += accel_x * dt
        self.est_vel['y'] += accel_y * dt
        self.est_vel['z'] += accel_z * dt

    def gps_callback(self, msg):
        """Low-frequency GPS update (EKF Update Step)"""
        if not self.initialized_gps:
            self.est_pos['x'] = msg.x
            self.est_pos['y'] = msg.y
            self.est_pos['z'] = msg.z
            self.initialized_gps = True
            # R8 FIX: Seed the real EKF with the first GPS fix
            if EKF_AVAILABLE:
                self.ekf = ExtendedKalmanFilter(initial_lat=msg.x, initial_lon=msg.y)
                self.get_logger().info(f'EKF seeded at ({msg.x:.6f}, {msg.y:.6f})')
            return
            
        if EKF_AVAILABLE and self.ekf:
            # Use the real EKF update step (covariance-weighted)
            self.ekf.update(msg.x, msg.y)
            ekf_lat, ekf_lon = self.ekf.get_state()
            self.est_pos['x'] = ekf_lat
            self.est_pos['y'] = ekf_lon
            self.est_pos['z'] = msg.z  # Altitude from GPS directly
        else:
            # Fallback: simple 20% trust complementary filter
            kalman_gain = 0.2
            self.est_pos['x'] += kalman_gain * (msg.x - self.est_pos['x'])
            self.est_pos['y'] += kalman_gain * (msg.y - self.est_pos['y'])
            self.est_pos['z'] += kalman_gain * (msg.z - self.est_pos['z'])
        
        # Publish the clean fused state
        fused_msg = Point()
        fused_msg.x = self.est_pos['x']
        fused_msg.y = self.est_pos['y']
        fused_msg.z = self.est_pos['z']
        
        self.state_pub.publish(fused_msg)

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
