import time
import math
import random

class DroneState:
    x, y, z = 0.0, 0.0, 0.0

class UltimateMonth3Mission:
    """
    Compresses Months 1, 2, and 3 into a single execution script.
    - Month 2: Waypoints & Obstacle Detection
    - Month 3: YOLO Object Detection, Return Home, Autonomous Landing
    """
    def __init__(self):
        self.state = DroneState()
        self.home = (0.0, 0.0, 0.0)
        self.waypoints = [(50.0, 50.0, 15.0), (100.0, -20.0, 20.0)]
        self.yolo_classes = ['Person', 'Bird', 'Another Drone', 'Landing Pad', 'Tree']
        
    def distance_2d(self, target_x, target_y):
        return math.sqrt((target_x - self.state.x)**2 + (target_y - self.state.y)**2)

    def takeoff(self):
        print("\n[PHASE 1] Autonomous Takeoff Sequence")
        while self.state.z < 10.0:
            self.state.z += 2.0
            print(f"   -> Climbing... Alt: {self.state.z}m")
            time.sleep(0.1)

    def run_yolo_vision(self):
        # Simulating Month 3: AI Object Detection
        if random.random() < 0.3:
            detected = random.choice(self.yolo_classes)
            confidence = random.uniform(85.0, 99.9)
            print(f"   >>> [YOLOv8 VISION] Detected: {detected} (Confidence: {confidence:.1f}%)")
            
            # Month 2: Obstacle Avoidance
            if detected in ['Bird', 'Another Drone', 'Tree']:
                print(f"   >>> [AVOIDANCE AI] Threat '{detected}' dead ahead! Executing evasive maneuver!")
                self.state.x += 10.0
                self.state.y -= 10.0
                time.sleep(0.5)

    def navigate_waypoints(self):
        print("\n[PHASE 2] Autonomous Waypoint Navigation & AI Vision")
        for i, wp in enumerate(self.waypoints):
            print(f"\n   --- Heading to Waypoint {i+1}: {wp} ---")
            self.state.z = wp[2]
            while self.distance_2d(wp[0], wp[1]) > 5.0:
                self.run_yolo_vision()
                
                dx, dy = wp[0] - self.state.x, wp[1] - self.state.y
                dist = self.distance_2d(wp[0], wp[1])
                
                if dist > 10.0:
                    self.state.x += (dx/dist) * 10.0
                    self.state.y += (dy/dist) * 10.0
                else:
                    self.state.x, self.state.y = wp[0], wp[1]
                    
                print(f"   -> Position: ({self.state.x:.1f}, {self.state.y:.1f}) | Distance: {self.distance_2d(wp[0], wp[1]):.1f}m")
                time.sleep(0.2)

    def return_home(self):
        print("\n[PHASE 3] Return-Home (RTL) AI Triggered")
        while self.distance_2d(self.home[0], self.home[1]) > 5.0:
            dx, dy = self.home[0] - self.state.x, self.home[1] - self.state.y
            dist = self.distance_2d(self.home[0], self.home[1])
            
            if dist > 15.0:
                self.state.x += (dx/dist) * 15.0
                self.state.y += (dy/dist) * 15.0
            else:
                self.state.x, self.state.y = self.home[0], self.home[1]
                
            print(f"   -> Flying back to Base... Distance: {dist:.1f}m")
            time.sleep(0.1)

    def autonomous_landing(self):
        print("\n[PHASE 4] Autonomous Landing Sequence")
        print("   >>> [YOLOv8 VISION] Detected: Landing Pad (Confidence: 98.7%)")
        print("   >>> [VISUAL SERVOING] Aligning descent vector with Landing Pad...")
        while self.state.z > 0.0:
            self.state.z -= 2.0
            if self.state.z < 0: self.state.z = 0.0
            print(f"   -> Descending... Alt: {self.state.z}m")
            time.sleep(0.2)
        print("\n>>> MISSION SUCCESS: Touchdown Confirmed. Drone Disarmed.")

    def execute_all(self):
        print("==========================================================")
        print(">>> COMPRESSING 3 MONTHS OF R&D INTO A SINGLE EXECUTION")
        print("==========================================================")
        self.takeoff()
        self.navigate_waypoints()
        self.return_home()
        self.autonomous_landing()

if __name__ == "__main__":
    mission = UltimateMonth3Mission()
    mission.execute_all()
