import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, Vector3
import random

class HardwareInterfaceNode(Node):
    """
    Layer 1: Sensors & Hardware (Perception Layer Input).
    This node acts as the HAL (Hardware Abstraction Layer). It connects to the physical 
    I2C, SPI, UART, and USB ports on the aircraft to pull raw sensory data and 
    broadcast it across the ROS 2 network.
    """
    def __init__(self):
        super().__init__('hardware_interface_node')
        
        # ROS 2 Publishers for all 4 sensor categories
        
        # 1. Vision Systems (High-res Optical & Infrared)
        # (In reality this would publish Image messages, using a Point for bounding box simulation)
        self.vision_pub = self.create_publisher(Point, '/sensors/vision/eo_ir', 10)
        
        # 2. Spatial Sensors (LiDAR & Radar for distance mapping)
        self.lidar_pub = self.create_publisher(Point, '/sensors/spatial/lidar_pointcloud', 10)
        
        # 3. Navigation (GNSS/GPS & IMU)
        self.gps_pub = self.create_publisher(Point, '/sensors/nav/gps', 10)
        self.imu_pub = self.create_publisher(Vector3, '/sensors/nav/imu', 10)
        
        # 4. Avionics Data (Pitot Tube Airspeed, Barometric Altimeter, Gyroscopes)
        self.avionics_pub = self.create_publisher(Point, '/sensors/avionics/flight_instruments', 10)
        
        # Timer to read from hardware at 100Hz
        self.timer = self.create_timer(0.01, self.read_hardware_sensors)
        self.get_logger().info('HardwareInterfaceNode initialized (Layer 1). Listening to physical ports...')

    def read_hardware_sensors(self):
        """Simulates polling physical serial/I2C buses on the aircraft."""
        
        # --- 3. Navigation ---
        # Mock GPS (GNSS) Reading
        gps_msg = Point()
        gps_msg.x = 47.3977 + random.uniform(-0.0001, 0.0001) # Lat
        gps_msg.y = 8.5455 + random.uniform(-0.0001, 0.0001)  # Lon
        gps_msg.z = 500.0 + random.uniform(-2.0, 2.0)         # GPS Alt
        self.gps_pub.publish(gps_msg)
        
        # Mock IMU (Acceleration) Reading
        imu_msg = Vector3()
        imu_msg.x = random.uniform(-0.5, 0.5)
        imu_msg.y = random.uniform(-0.5, 0.5)
        imu_msg.z = 9.81 + random.uniform(-0.1, 0.1) # Gravity
        self.imu_pub.publish(imu_msg)
        
        # The 100Hz loop is too fast for logging, we'll selectively log every 100 frames
        if random.random() < 0.01:
            
            # --- 1. Vision Systems ---
            # Simulate EO/IR detecting a runway
            vision_msg = Point(x=320.0, y=240.0, z=1.0) # 1.0 = Runway confidence
            self.vision_pub.publish(vision_msg)
            self.get_logger().info("[VISION] Optical/IR Camera feed published. Runway detected.")
            
            # --- 2. Spatial Sensors ---
            # Simulate LiDAR scanning terrain
            lidar_msg = Point(x=0.0, y=0.0, z=150.0) # 150m clearance below
            self.lidar_pub.publish(lidar_msg)
            self.get_logger().info("[SPATIAL] LiDAR pointcloud published. Terrain clearance: 150m.")
            
            # --- 4. Avionics Data ---
            # Simulate Pitot tube dynamic pressure to get Indicated Airspeed (IAS)
            avionics_msg = Point()
            avionics_msg.x = 120.0 # IAS in knots
            avionics_msg.y = 500.0 # Barometric altitude
            avionics_msg.z = 0.0   # Gyro slip
            self.avionics_pub.publish(avionics_msg)
            self.get_logger().info(f"[AVIONICS] Pitot Tube Airspeed: {avionics_msg.x} knots. Barometer: {avionics_msg.y} ft.")

def main(args=None):
    rclpy.init(args=args)
    node = HardwareInterfaceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
