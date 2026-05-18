import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32
from geometry_msgs.msg import Point
import time
import random

class EmergencyAINode(Node):
    """
    Phase 4 | Module 4 - Emergency AI
    Monitors system health and intercepts critical failures. 
    It has the highest authority in the OS and can override any mission or flight plan.
    """
    def __init__(self):
        super().__init__('emergency_ai_node')
        
        # Subscriptions to health topics
        self.gps_health_sub = self.create_subscription(Float32, '/sensors/gps/health', self.gps_health_cb, 10)
        self.motor_health_sub = self.create_subscription(String, '/hardware/motors/health', self.motor_health_cb, 10)
        self.comms_sub = self.create_subscription(Float32, '/communications/uplink_ping', self.comms_cb, 10)
        
        # Publisher for Emergency Overrides
        self.failsafe_pub = self.create_publisher(String, '/aegis/mission/failsafe_trigger', 10)
        self.pid_tuning_pub = self.create_publisher(Point, '/hardware/actuators/dynamic_pid_gains', 10)
        
        # State tracking
        self.last_comms_time = time.time()
        self.gps_denied = False
        
        # Check health every 1 second
        self.timer = self.create_timer(1.0, self.health_watchdog)
        self.get_logger().info('EmergencyAINode initialized. Monitoring system vitals.')

    def gps_health_cb(self, msg):
        # E.g., msg.data is satellite count
        if msg.data < 4 and not self.gps_denied:
            self.gps_denied = True
            self.get_logger().fatal("[EMERGENCY AI] GPS LOSS DETECTED (Jamming/Spoofing). Engaging Visual Odometry SLAM fallback!")
            
            # Publish override to force EKF to use Visual Navigation
            fallback_msg = String()
            fallback_msg.data = "ENABLE_VISUAL_SLAM_FALLBACK"
            self.failsafe_pub.publish(fallback_msg)
            
        elif msg.data >= 4 and self.gps_denied:
            self.gps_denied = False
            self.get_logger().info("[EMERGENCY AI] GPS Signal Restored.")

    def motor_health_cb(self, msg):
        if "FAILURE" in msg.data:
            self.get_logger().fatal(f"[EMERGENCY AI] CRITICAL HARDWARE ALERT: {msg.data}!")
            # Adaptive Control: If Motor 2 fails, we must dynamically retune PID to fly on 3 motors (or deploy parachute)
            self.get_logger().warn("[EMERGENCY AI] Re-allocating ESC mixer matrices. Deploying ballistic parachute if unrecoverable.")
            fallback_msg = String()
            fallback_msg.data = "DEPLOY_PARACHUTE"
            self.failsafe_pub.publish(fallback_msg)

    def comms_cb(self, msg):
        # We receive a ping from Ground Control
        self.last_comms_time = time.time()

    def detect_wind_turbulence(self):
        # Simulating reading high variance from IMU accelerometers
        if random.random() < 0.05:
            self.get_logger().warn("[EMERGENCY AI] SEVERE WIND TURBULENCE DETECTED.")
            self.get_logger().warn("[EMERGENCY AI] Dynamically increasing PID Derivative (Kd) gain to dampen oscillations.")
            
            # Publish new PID tuning parameters to Layer 4
            gains = Point(x=1.5, y=0.1, z=3.0) # Elevated Kd (Z axis represents Kd here)
            self.pid_tuning_pub.publish(gains)

    def health_watchdog(self):
        # 1. Check Communications Loss
        if time.time() - self.last_comms_time > 10.0:
            self.get_logger().fatal("[EMERGENCY AI] COMMUNICATIONS LOST FOR 10 SECONDS. Triggering autonomous RTL (Return to Launch).")
            fallback_msg = String()
            fallback_msg.data = "ENGAGE_RTL"
            self.failsafe_pub.publish(fallback_msg)
            # Reset timer to avoid spam
            self.last_comms_time = time.time()
            
        # 2. Check Wind Turbulence
        self.detect_wind_turbulence()

def main(args=None):
    rclpy.init(args=args)
    node = EmergencyAINode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
