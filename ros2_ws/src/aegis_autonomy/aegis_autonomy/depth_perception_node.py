import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Bool, Float32
import struct
import sys
import os

# Import the embedded driver for calculations
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../src')))
try:
    from embedded.realsense_d435_driver import RealSenseD435Driver
except ImportError:
    RealSenseD435Driver = None


class DepthPerceptionNode(Node):
    """
    ROS 2 Node that subscribes to raw PointCloud2 messages (e.g. from librealsense d435)
    and performs a 3D bounding box collision check in real-time, publishing warnings.
    """
    def __init__(self):
        super().__init__('depth_perception_node')
        
        # Subscribes to RealSense depth camera pointcloud topic
        self.pc_sub = self.create_subscription(
            PointCloud2,
            '/camera/depth/color/points',
            self.pointcloud_callback,
            10
        )
        
        # Publishes whether a collision is detected in the safety envelope
        self.collision_pub = self.create_publisher(Bool, '/aegis/safety/collision_warning', 10)
        self.distance_pub = self.create_publisher(Float32, '/aegis/safety/closest_obstacle_dist', 10)
        
        # Instantiate Driver logic for calculations
        self.driver = RealSenseD435Driver()
        self.get_logger().info('DepthPerceptionNode (RealSense D435 PointCloud2) initialized.')

    def pointcloud_callback(self, msg):
        """Processes PointCloud2 message, extracts points, and checks for collisions."""
        points = []
        
        # Parse PointCloud2 binary data
        # PointCloud2 has fields: x, y, z as float32
        # We can unpack the bytes directly for performance
        try:
            # Get offsets for x, y, z
            x_offset, y_offset, z_offset = 0, 4, 8
            for field in msg.fields:
                if field.name == 'x':
                    x_offset = field.offset
                elif field.name == 'y':
                    y_offset = field.offset
                elif field.name == 'z':
                    z_offset = field.offset

            # Unpack points
            fmt = f'<{len(msg.data)}B'
            data_bytes = struct.unpack(fmt, msg.data)
            point_step = msg.point_step
            row_step = msg.row_step

            # Unpack a subset of points for computational efficiency
            # (Sampling 1 out of every 10 points to run at 30Hz without lag)
            for i in range(0, len(msg.data), point_step * 10):
                offset = i
                # Read x, y, z floats
                x = struct.unpack('<f', bytes(data_bytes[offset + x_offset : offset + x_offset + 4]))[0]
                y = struct.unpack('<f', bytes(data_bytes[offset + y_offset : offset + y_offset + 4]))[0]
                z = struct.unpack('<f', bytes(data_bytes[offset + z_offset : offset + z_offset + 4]))[0]
                
                # Check for valid points
                if not (math.isnan(x) or math.isnan(y) or math.isnan(z)):
                    points.append([x, y, z])
        except Exception as e:
            # Fallback to driver's internal simulated points if parsing fails
            points = self.driver.get_pointcloud()

        # Perform 3D Bounding Box Collision Check
        collision = False
        closest_dist = None
        if len(points) > 0:
            collision, closest_dist = self.driver.check_collision(
                points,
                max_distance_m=12.0,
                box_width_m=2.5,
                box_height_m=2.0
            )

        # Publish Boolean warning
        warn_msg = Bool()
        warn_msg.data = collision
        self.collision_pub.publish(warn_msg)

        # Publish Float proximity
        dist_msg = Float32()
        dist_msg.data = closest_dist if closest_dist is not None else -1.0
        self.distance_pub.publish(dist_msg)

        if collision:
            self.get_logger().warn(
                f"[COLLISION WARNING] Obstacle detected! Distance: {closest_dist:.2f}m. Sending abort signal."
            )


# Import math here just in case
import math

def main(args=None):
    rclpy.init(args=args)
    node = DepthPerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
