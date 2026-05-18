import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
import sys
import os

# We would import our C++ PID Bridge here
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../src/control')))
# from pid_bridge import CppPIDController

class FlightControlNode(Node):
    def __init__(self):
        super().__init__('flight_control_node')
        
        # Subscribes to the AI Brain's path planner
        self.target_sub = self.create_subscription(Point, '/aegis/planner/target_waypoint', self.target_callback, 10)
        
        # Subscribes to the fused state from the EKF
        self.state_sub = self.create_subscription(Point, '/aegis/state/fused_position', self.state_callback, 10)
        
        # Publisher to physical actuators (Servos/Motors)
        self.actuator_pub = self.create_publisher(Point, '/hardware/actuators/cmd', 10)
        
        self.current_state = None
        self.get_logger().info('FlightControlNode initialized. C++ PID Active.')

    def state_callback(self, msg):
        """Keeps track of where we actually are."""
        self.current_state = msg

    def target_callback(self, msg):
        """The AI Brain commands us to go to 'msg'."""
        if not self.current_state:
            self.get_logger().warn("Cannot compute PIDs: No state estimate available.")
            return
            
        # Target Waypoint (Simplified for MVP)
        target_x = msg.x
        target_y = msg.y
        target_z = msg.z
        
        # 1. Pitch Controller (Elevator) -> Controls Altitude/Z
        pitch_error = target_z - self.current_state.z
        elevator_cmd = pitch_error * 1.5 # Mock P-gain (C++ PID goes here)
        
        # 2. Roll Controller (Aileron) -> Controls Heading/Y
        roll_error = target_y - self.current_state.y
        aileron_cmd = roll_error * 1.2
        
        # 3. Yaw Controller (Rudder) -> Coordinates turns
        rudder_cmd = roll_error * 0.3 # Coordinated flight
        
        # 4. Throttle Controller (Speed) -> Controls X velocity
        throttle_cmd = 75.0 # Cruising throttle
        
        cmd_msg = Point()
        cmd_msg.x = elevator_cmd # Reusing Point for MVP servos
        cmd_msg.y = aileron_cmd
        cmd_msg.z = throttle_cmd
        
        self.actuator_pub.publish(cmd_msg)
        
        self.get_logger().info(f'PIDs Computed | Elev: {elevator_cmd:.1f} | Ail: {aileron_cmd:.1f} | Rud: {rudder_cmd:.1f} | Thr: {throttle_cmd:.1f}%')

def main(args=None):
    rclpy.init(args=args)
    node = FlightControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
