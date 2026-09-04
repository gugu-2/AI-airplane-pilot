import rclpy
from rclpy.node import Node
import time
import random
import json

class MissionManagementNode(Node):
    """
    Layer 6: Mission Management.
    Handles high-level fleet logistics, passenger comfort, and strict airspace regulations.
    This sits above the cognitive and flight control layers.
    """
    def __init__(self):
        super().__init__('mission_management_node')
        
        # In a real setup, these would be subscribed to specific ROS 2 topics
        self.max_altitude_limit = 10000.0 # Airspace regulation constraint
        self.geofence_bounds = {"min_lat": 47.0, "max_lat": 48.0, "min_lon": 8.0, "max_lon": 9.0}
        
        # Passenger Comfort Metrics
        self.cabin_pressure_psi = 14.7 # Sea level pressure
        self.cabin_temp_c = 22.0
        
        # Flight Schedule
        self.flight_manifest = {
            "flight_id": "AEGIS-77",
            "departure_time": time.time() + 60, # Departure in 60 seconds
            "passengers_onboard": 4,
            "status": "BOARDING"
        }
        
        # Timer to run mission checks every 2 seconds
        self.timer = self.create_timer(2.0, self.evaluate_mission_status)
        self.get_logger().info('MissionManagementNode initialized (Layer 6).')

    def check_airspace_rules(self, current_lat, current_lon, current_alt):
        """Enforce strict FAA/DGCA airspace regulations."""
        violations = []
        if current_alt > self.max_altitude_limit:
            violations.append(f"ALTITUDE VIOLATION: {current_alt} exceeds limit of {self.max_altitude_limit}")
            
        if not (self.geofence_bounds["min_lat"] <= current_lat <= self.geofence_bounds["max_lat"]):
            violations.append(f"GEOFENCE VIOLATION: Latitude {current_lat} out of bounds.")
            
        if violations:
            for v in violations:
                self.get_logger().error(f"[AIRSPACE] {v}")
            return False
        return True

    def monitor_passenger_comfort(self):
        """Simulate monitoring cabin environment for human passengers."""
        # Randomly fluctuate metrics
        self.cabin_pressure_psi += random.uniform(-0.1, 0.1)
        self.cabin_temp_c += random.uniform(-0.5, 0.5)
        
        if self.cabin_pressure_psi < 10.0:
            self.get_logger().fatal("[PASSENGER SAFETY] Cabin depressurization detected! Initiating emergency descent.")
            # Here it would publish an emergency override to the RL Cognitive Node
        elif self.cabin_temp_c > 28.0 or self.cabin_temp_c < 18.0:
            self.get_logger().warn(f"[PASSENGER COMFORT] Adjusting HVAC. Temp is {self.cabin_temp_c:.1f}C")
        else:
            self.get_logger().info(f"[PASSENGER COMFORT] Nominal. Temp: {self.cabin_temp_c:.1f}C, Pressure: {self.cabin_pressure_psi:.1f} PSI")

    def evaluate_mission_status(self):
        """High-level state machine for the entire flight."""
        current_time = time.time()
        
        if self.flight_manifest["status"] == "BOARDING":
            if current_time >= self.flight_manifest["departure_time"]:
                self.flight_manifest["status"] = "TAXIING"
                self.get_logger().info("[SCHEDULING] Boarding complete. Commencing taxi.")
            else:
                time_left = int(self.flight_manifest["departure_time"] - current_time)
                self.get_logger().info(f"[SCHEDULING] Boarding. Departure in {time_left}s. Passengers: {self.flight_manifest['passengers_onboard']}")
                
        elif self.flight_manifest["status"] == "TAXIING":
            self.get_logger().info("[SCHEDULING] Taxiing to runway. Awaiting ATC clearance.")
            # In real life, wait for NLP Agent to confirm takeoff clearance
            self.flight_manifest["status"] = "IN_FLIGHT"
            
        elif self.flight_manifest["status"] == "IN_FLIGHT":
            # Mock current state
            mock_lat = 47.5
            mock_lon = 8.5
            mock_alt = 500.0
            
            # 1. Enforce Airspace Rules
            self.check_airspace_rules(mock_lat, mock_lon, mock_alt)
            
            # 2. Monitor Passenger Comfort
            self.monitor_passenger_comfort()

def main(args=None):
    rclpy.init(args=args)
    node = MissionManagementNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
