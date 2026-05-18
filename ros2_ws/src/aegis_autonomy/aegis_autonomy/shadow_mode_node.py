import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
import time

class ShadowModeNode(Node):
    """
    Phase 3: Shadow Mode Evaluator.
    Runs in the background on a human-piloted aircraft. 
    It compares the AI's intended actuator commands against the human pilot's actual 
    physical inputs to measure the AI's safety and reliability without risking the aircraft.
    """
    def __init__(self):
        super().__init__('shadow_mode_node')
        
        # Subscribe to what the AI *wants* to do
        self.ai_sub = self.create_subscription(Point, '/hardware/actuators/cmd', self.ai_callback, 10)
        
        # Subscribe to what the Human is *actually* doing on the yoke
        self.human_sub = self.create_subscription(Point, '/hardware/human_pilot/yoke', self.human_callback, 10)
        
        self.ai_latest_cmd = None
        self.human_latest_cmd = None
        
        self.total_evaluations = 0
        self.divergence_count = 0
        
        # Compare them every 0.5 seconds
        self.timer = self.create_timer(0.5, self.evaluate_shadow_mode)
        
        self.get_logger().info('ShadowModeNode initialized. Aircraft is in READ-ONLY Mode.')

    def ai_callback(self, msg):
        self.ai_latest_cmd = msg

    def human_callback(self, msg):
        self.human_latest_cmd = msg

    def evaluate_shadow_mode(self):
        if not self.ai_latest_cmd or not self.human_latest_cmd:
            return
            
        self.total_evaluations += 1
        
        # Compare Elevator (Pitch) Commands
        ai_pitch = self.ai_latest_cmd.x
        human_pitch = self.human_latest_cmd.x
        
        # If AI diverges from human by more than 10 degrees, flag it as a mistake
        if abs(ai_pitch - human_pitch) > 10.0:
            self.divergence_count += 1
            self.get_logger().warn(
                f"[SHADOW MODE EXCEPTION] AI intended {ai_pitch:.1f} but Human applied {human_pitch:.1f}. "
                f"(Error Rate: {(self.divergence_count/self.total_evaluations)*100:.1f}%)"
            )
        else:
            self.get_logger().info("[SHADOW MODE] AI and Human Pilot in consensus.")

def main(args=None):
    rclpy.init(args=args)
    node = ShadowModeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
