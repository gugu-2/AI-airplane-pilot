import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point

class EnvelopeProtectionNode(Node):
    """
    Critical Safety Feature: Envelope Protection.
    This is a hardcoded, deterministic (non-AI) firewall. 
    It intercepts the AI's requested servo commands and rigidly enforces 
    aerodynamic limits to prevent the AI from stalling or over-stressing the airframe.
    """
    def __init__(self):
        super().__init__('envelope_protection_node')
        
        # Subscribe to what the AI/FBW *wants* to actuate
        self.raw_cmd_sub = self.create_subscription(Point, '/hardware/actuators/cmd_raw', self.raw_cmd_callback, 10)
        
        # Publisher to the actual physical servos
        self.safe_cmd_pub = self.create_publisher(Point, '/hardware/actuators/cmd_safe', 10)
        
        # Hardcoded Aerodynamic Flight Envelope Limits (e.g., Cessna 172 limits)
        self.MAX_PITCH = 15.0  # Degrees up/down
        self.MAX_ROLL = 30.0   # Degrees left/right
        
        self.get_logger().info('EnvelopeProtectionNode initialized. Hardcoded Safety Limits Active.')

    def raw_cmd_callback(self, msg):
        """Intercepts the raw command and clamps it to the flight envelope."""
        
        safe_msg = Point()
        
        # 1. Protect Pitch (Elevator)
        raw_pitch = msg.x
        if raw_pitch > self.MAX_PITCH:
            self.get_logger().error(f"[ENVELOPE PROTECTION] AI commanded pitch {raw_pitch:.1f} > MAX {self.MAX_PITCH}. Clamping.")
            safe_msg.x = self.MAX_PITCH
        elif raw_pitch < -self.MAX_PITCH:
            self.get_logger().error(f"[ENVELOPE PROTECTION] AI commanded pitch {raw_pitch:.1f} < MIN -{self.MAX_PITCH}. Clamping.")
            safe_msg.x = -self.MAX_PITCH
        else:
            safe_msg.x = raw_pitch
            
        # 2. Protect Roll (Aileron)
        raw_roll = msg.y
        if raw_roll > self.MAX_ROLL:
            self.get_logger().error(f"[ENVELOPE PROTECTION] AI commanded roll {raw_roll:.1f} > MAX {self.MAX_ROLL}. Clamping.")
            safe_msg.y = self.MAX_ROLL
        elif raw_roll < -self.MAX_ROLL:
            self.get_logger().error(f"[ENVELOPE PROTECTION] AI commanded roll {raw_roll:.1f} < MIN -{self.MAX_ROLL}. Clamping.")
            safe_msg.y = -self.MAX_ROLL
        else:
            safe_msg.y = raw_roll
            
        safe_msg.z = msg.z # Throttle bypasses this specific check in MVP
        
        # Publish the legally safe command to the physical servos
        self.safe_cmd_pub.publish(safe_msg)

def main(args=None):
    rclpy.init(args=args)
    node = EnvelopeProtectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
