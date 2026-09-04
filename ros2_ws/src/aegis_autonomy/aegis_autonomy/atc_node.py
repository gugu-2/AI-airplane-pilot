import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import sys
import os

# Import our NLP Engine
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../src/cognitive')))
from atc_agent import ATCNaturalLanguageProcessor

class ATCNode(Node):
    """
    Cognitive Layer: NLP Air Traffic Control Node.
    Listens to simulated VHF radio audio (text transcripts in this MVP),
    processes the aviation intent, and broadcasts commands to the flight planner.
    """
    def __init__(self):
        super().__init__('atc_node')
        
        self.nlp_engine = ATCNaturalLanguageProcessor(callsign="Aegis 1")
        
        # Subscribe to incoming radio transcripts
        self.radio_sub = self.create_subscription(String, '/communications/vhf_radio/rx', self.radio_callback, 10)
        
        # Publisher for mission overrides (e.g., abort landing)
        self.override_pub = self.create_publisher(String, '/aegis/mission/atc_override', 10)
        
        self.get_logger().info('ATC NLP Node initialized. Listening on VHF Radio...')

    def radio_callback(self, msg):
        transcript = msg.data
        self.get_logger().info(f"[VHF RX] {transcript}")
        
        # Process the language
        parsed = self.nlp_engine.process_audio_transcript(transcript)
        
        if parsed["intent"] != "IGNORE":
            readback = self.nlp_engine.generate_readback(parsed)
            self.get_logger().info(f"[VHF TX] {readback}")
            
            # Broadcast critical intents to the RL Planner
            if parsed["intent"] in ["ABORT_LANDING", "ALTITUDE_CHANGE", "HOLD_POSITION"]:
                override_msg = String()
                override_msg.data = parsed["intent"]
                self.override_pub.publish(override_msg)
                self.get_logger().warn(f"[ATC] Broadcasting Override: {parsed['intent']}")

def main(args=None):
    rclpy.init(args=args)
    node = ATCNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
