import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
from sensor_msgs.msg import Image
from mavros_msgs.srv import CommandBool, SetMode, CommandTOL
from mavros_msgs.msg import State
import cv2
import numpy as np
from cv_bridge import CvBridge

class MavrosHoopMission(Node):
    """
    Phase 2 Integration: PX4 + Gazebo/AirSim + MAVROS.
    Commands a simulated drone to take off, use OpenCV to find a hoop, 
    fly through it, and land autonomously.
    """
    def __init__(self):
        super().__init__('mavros_hoop_mission')
        
        self.state = State()
        self.bridge = CvBridge()
        self.hoop_detected = False
        self.hoop_center = (0, 0)
        self.image_center = (320, 240) # Assuming 640x480 camera
        
        # MAVROS Subscribers
        self.state_sub = self.create_subscription(State, '/mavros/state', self.state_cb, 10)
        self.image_sub = self.create_subscription(Image, '/camera/rgb/image_raw', self.image_cb, 10)
        
        # MAVROS Publishers
        self.local_pos_pub = self.create_publisher(PoseStamped, '/mavros/setpoint_position/local', 10)
        self.vel_pub = self.create_publisher(Twist, '/mavros/setpoint_velocity/cmd_vel_unstamped', 10)
        
        # MAVROS Services
        self.arming_client = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.set_mode_client = self.create_client(SetMode, '/mavros/set_mode')
        self.takeoff_client = self.create_client(CommandTOL, '/mavros/cmd/takeoff')
        self.land_client = self.create_client(CommandTOL, '/mavros/cmd/land')
        
        self.mission_state = "INIT"
        self.timer = self.create_timer(0.1, self.mission_loop)
        self.get_logger().info("MAVROS Hoop Mission Node Initialized.")

    def state_cb(self, msg):
        self.state = msg

    def image_cb(self, msg):
        """Computer Vision: Detect a Red Hoop using OpenCV"""
        if self.mission_state not in ["SEARCHING_HOOP", "FLYING_THROUGH_HOOP"]:
            return
            
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
            
            # Threshold for Red Hoop
            lower_red = np.array([0, 120, 70])
            upper_red = np.array([10, 255, 255])
            mask = cv2.inRange(hsv, lower_red, upper_red)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Find largest contour (assuming it's the hoop)
                largest_contour = max(contours, key=cv2.contourArea)
                if cv2.contourArea(largest_contour) > 500:
                    (x, y), radius = cv2.minEnclosingCircle(largest_contour)
                    self.hoop_center = (int(x), int(y))
                    self.hoop_detected = True
                    return
                    
            self.hoop_detected = False
        except Exception as e:
            self.get_logger().error(f"CV2 Error: {e}")

    def arm_and_takeoff(self, target_alt=3.0):
        self.get_logger().info("Arming motors...")
        while not self.state.armed:
            req = CommandBool.Request()
            req.value = True
            self.arming_client.call_async(req)
            
        self.get_logger().info("Taking off...")
        req = SetMode.Request()
        req.custom_mode = 'OFFBOARD'
        self.set_mode_client.call_async(req)
        
        # Publish initial setpoints required for OFFBOARD mode
        pose = PoseStamped()
        pose.pose.position.z = target_alt
        for _ in range(100):
            self.local_pos_pub.publish(pose)
            
        self.mission_state = "SEARCHING_HOOP"

    def mission_loop(self):
        if self.mission_state == "INIT":
            if self.state.connected:
                self.arm_and_takeoff()
                
        elif self.mission_state == "SEARCHING_HOOP":
            if self.hoop_detected:
                self.get_logger().info(f"Hoop locked at {self.hoop_center}. Commencing approach.")
                self.mission_state = "FLYING_THROUGH_HOOP"
            else:
                # Hover and rotate to search
                vel_msg = Twist()
                vel_msg.angular.z = 0.5
                self.vel_pub.publish(vel_msg)
                
        elif self.mission_state == "FLYING_THROUGH_HOOP":
            if not self.hoop_detected:
                self.get_logger().info("Hoop lost or passed through. Initiating Landing.")
                self.mission_state = "LANDING"
                return
                
            # Visual Servoing: Steer drone based on hoop center error
            error_x = self.image_center[0] - self.hoop_center[0]
            error_y = self.image_center[1] - self.hoop_center[1]
            
            vel_msg = Twist()
            # Move forward towards the hoop
            vel_msg.linear.x = 1.0 
            # Correct horizontal drift
            vel_msg.linear.y = float(error_x) * 0.005 
            # Correct altitude drift
            vel_msg.linear.z = float(error_y) * 0.005 
            
            self.vel_pub.publish(vel_msg)
            
        elif self.mission_state == "LANDING":
            self.get_logger().info("Autonomous Landing sequence engaged.")
            req = CommandTOL.Request()
            self.land_client.call_async(req)
            self.mission_state = "COMPLETE"

def main(args=None):
    rclpy.init(args=args)
    node = MavrosHoopMission()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
