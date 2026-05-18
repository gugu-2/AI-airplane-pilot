import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, Vector3
import random
import sys
import os

# Add the main src directory to python path to import embedded drivers
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../src')))

try:
    from embedded.imu_driver import IMUDriver
    from embedded.gps_driver import GPSDriver
    from embedded.lidar_driver import LidarDriver
    from embedded.pitot_tube_driver import PitotTubeDriver
except ImportError as e:
    print(f"Warning: Embedded drivers not found: {e}")
    IMUDriver = GPSDriver = LidarDriver = PitotTubeDriver = None

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
        
        # R9 FIX: Accept home position as ROS2 parameters instead of hardcoding Zurich
        self.declare_parameter('home_lat', 47.3977)
        self.declare_parameter('home_lon', 8.5455)
        self.home_lat = self.get_parameter('home_lat').value
        self.home_lon = self.get_parameter('home_lon').value
        self.get_logger().info(f'Home position: ({self.home_lat}, {self.home_lon})')

        # A6 FIX: Restore the 100Hz sensor polling timer (was accidentally removed in Fix R9)
        self.timer = self.create_timer(0.01, self.read_hardware_sensors)
        self.get_logger().info('HardwareInterfaceNode initialized. Polling at 100Hz.')

        self.imu = self._try_init_driver(IMUDriver, "IMU", bus_num=1, address=0x68)
        self.gps = self._try_init_driver(GPSDriver, "GPS", port='/dev/ttyTHS1', baudrate=9600)
        self.lidar = self._try_init_driver(LidarDriver, "LiDAR", port='/dev/ttyS0', baudrate=115200)
        self.pitot = self._try_init_driver(PitotTubeDriver, "Pitot Tube", bus_num=1, address=0x28)

        self.get_logger().info('HardwareInterfaceNode initialized (Layer 1). Listening to physical ports...')

    def _try_init_driver(self, driver_class, name, **kwargs):
        if driver_class is None:
            return None
        try:
            driver = driver_class(**kwargs)
            self.get_logger().info(f"[HARDWARE] {name} initialized successfully.")
            return driver
        except Exception as e:
            self.get_logger().warn(f"[SIMULATION FALLBACK] Failed to init {name}: {e}. Using mock data.")
            return None

    def read_hardware_sensors(self):
        """Poll physical serial/I2C buses, falling back to simulation if uninitialized."""
        
        # --- 3. Navigation ---
        gps_msg = Point()
        if self.gps:
            pos = self.gps.read_position()
            if pos:
                gps_msg.x, gps_msg.y, gps_msg.z = pos
            else:
                # Keep previous if no fix
                pass 
        else:
            # R9 FIX: Use the configurable home position, not hardcoded Zurich
            gps_msg.x = self.home_lat + random.uniform(-0.0001, 0.0001)
            gps_msg.y = self.home_lon + random.uniform(-0.0001, 0.0001)
            gps_msg.z = 500.0 + random.uniform(-2.0, 2.0)
        self.gps_pub.publish(gps_msg)
        
        imu_msg = Vector3()
        if self.imu:
            ax, ay, az = self.imu.read_acceleration()
            imu_msg.x, imu_msg.y, imu_msg.z = ax, ay, az
        else:
            imu_msg.x = random.uniform(-0.5, 0.5)
            imu_msg.y = random.uniform(-0.5, 0.5)
            imu_msg.z = 9.81 + random.uniform(-0.1, 0.1)
        self.imu_pub.publish(imu_msg)
        
        # The 100Hz loop is too fast for logging, selectively log every 100 frames
        if random.random() < 0.01:
            
            # --- 1. Vision Systems ---
            vision_msg = Point(x=320.0, y=240.0, z=1.0)
            self.vision_pub.publish(vision_msg)
            
            # --- 2. Spatial Sensors ---
            lidar_msg = Point()
            lidar_msg.x = 0.0
            lidar_msg.y = 0.0
            if self.lidar:
                dist = self.lidar.read_distance()
                lidar_msg.z = dist if dist else 150.0
            else:
                lidar_msg.z = 150.0 + random.uniform(-0.5, 0.5)
            self.lidar_pub.publish(lidar_msg)
            
            # --- 4. Avionics Data ---
            avionics_msg = Point()
            if self.pitot:
                ias = self.pitot.read_airspeed()
                avionics_msg.x = ias if ias is not None else 120.0
            else:
                avionics_msg.x = 120.0 + random.uniform(-1.0, 1.0)
                
            avionics_msg.y = 500.0 # Barometric altitude
            avionics_msg.z = 0.0   # Gyro slip
            self.avionics_pub.publish(avionics_msg)
            
            # Log for debugging
            self.get_logger().info(f"[SENSORS] GPS:({gps_msg.x:.4f}, {gps_msg.y:.4f}), IMU_Z:{imu_msg.z:.2f}, Lidar_Z:{lidar_msg.z:.1f}, IAS:{avionics_msg.x:.1f}")

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
