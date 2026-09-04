import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
import torch
import torch.nn as nn
import numpy as np
import random

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../src/cognitive')))
from rl_models import FlightRLAgent

# =====================================================================
# ROS 2 Cognitive Planner Node
# =====================================================================
class CognitivePlannerNode(Node):
    def __init__(self):
        super().__init__('cognitive_planner_node')
        
        # Publisher: Send desired waypoints to the C++ Flight Controller
        self.target_pub = self.create_publisher(Point, '/aegis/planner/target_waypoint', 10)
        
        # Subscriber: Listen to the EKF Fused State
        self.state_sub = self.create_subscription(Point, '/aegis/state/fused_position', self.state_callback, 10)
        
        # Initialize the RL Agent
        self.rl_agent = FlightRLAgent()
        
        # In a real scenario, we would load pre-trained weights from the cloud:
        # self.rl_agent.load_state_dict(torch.load('flight_rl_model.pth'))
        self.rl_agent.eval() # Set to inference mode
        
        self.current_state = None
        
        # Run the cognitive decision loop at 10Hz
        self.timer = self.create_timer(0.1, self.decision_loop)
        
        self.get_logger().info('CognitivePlannerNode initialized. RL Agent Active.')

    def state_callback(self, msg):
        self.current_state = msg

    def decision_loop(self):
        if not self.current_state:
            return

        # 1. Construct the State Vector for the Neural Network
        # [current_x, current_y, current_alt, distance_to_nearest_obstacle]
        # (Mocking obstacle distance to 50 meters for this MVP)
        state_tensor = torch.tensor([
            self.current_state.x, 
            self.current_state.y, 
            self.current_state.z, 
            50.0 
        ], dtype=torch.float32)

        # 2. RL Agent Inference: Predict optimal action (Q-Values)
        with torch.no_grad():
            q_values = self.rl_agent(state_tensor)
            best_action = torch.argmax(q_values).item()

        # 3. Translate RL Action into a Physical Waypoint
        # Action Map: 0=Maintain, 1=Climb, 2=Descend, 3=Left, 4=Right
        target_msg = Point()
        target_msg.x = self.current_state.x
        target_msg.y = self.current_state.y
        target_msg.z = self.current_state.z
        
        if best_action == 1:
            target_msg.z += 10.0
            action_desc = "CLIMB"
        elif best_action == 2:
            target_msg.z -= 10.0
            action_desc = "DESCEND"
        elif best_action == 3:
            target_msg.y -= 10.0
            action_desc = "BANK LEFT"
        elif best_action == 4:
            target_msg.y += 10.0
            action_desc = "BANK RIGHT"
        else:
            action_desc = "MAINTAIN COURSE"
            target_msg.x += 10.0 # Just move forward

        # 4. Publish the target for the C++ PID Controller to execute
        self.target_pub.publish(target_msg)
        
        # Don't spam the logs at 10Hz, print occasionally
        if random.random() > 0.9:
            self.get_logger().info(f'[RL Inference] Action: {action_desc} -> New Target: Z={target_msg.z:.1f}')

def main(args=None):
    rclpy.init(args=args)
    node = CognitivePlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
