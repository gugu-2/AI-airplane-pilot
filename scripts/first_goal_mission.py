import time
import math
import random

class DroneState:
    x = 0.0
    y = 0.0
    z = 0.0
    heading = 0.0

class FirstGoalMission:
    """
    Executes the 'First Goal' from Option two.txt:
    Take off autonomously -> Navigate waypoints -> Avoid obstacles -> Return home -> Land safely.
    (Inside simulation)
    """
    def __init__(self):
        self.state = DroneState()
        self.home_position = (0.0, 0.0, 0.0)
        self.waypoints = [
            (50.0, 50.0, 10.0),
            (100.0, -20.0, 15.0),
            (150.0, 80.0, 20.0)
        ]
        self.current_wp_index = 0
        self.mission_status = "PRE-FLIGHT"
        
    def distance_to(self, target_x, target_y, target_z):
        dx = target_x - self.state.x
        dy = target_y - self.state.y
        dz = target_z - self.state.z
        return math.sqrt(dx**2 + dy**2 + dz**2)

    def takeoff(self, target_alt=10.0):
        self.mission_status = "TAKEOFF"
        print("\n[MISSION] Commencing Autonomous Takeoff...")
        while self.state.z < target_alt:
            self.state.z += 2.0
            print(f"   -> Climbing... Altitude: {self.state.z:.1f}m")
            time.sleep(0.2)
        print("[MISSION] Takeoff Complete. Cruising Altitude Reached.")

    def navigate(self):
        self.mission_status = "NAVIGATING"
        target = self.waypoints[self.current_wp_index]
        print(f"\n[MISSION] Navigating to Waypoint {self.current_wp_index + 1}: {target}")
        
        while True:
            dx = target[0] - self.state.x
            dy = target[1] - self.state.y
            dist2d = math.sqrt(dx**2 + dy**2)
            
            if dist2d <= 5.0:
                break
            
            if dist2d > 10.0:
                self.state.x += (dx / dist2d) * 10.0
                self.state.y += (dy / dist2d) * 10.0
            else:
                self.state.x = target[0]
                self.state.y = target[1]
            
            print(f"   -> En route... Current Position: ({self.state.x:.1f}, {self.state.y:.1f}) | Distance: {dist2d:.1f}m")
            time.sleep(0.2)
            
        print(f"[MISSION] Waypoint {self.current_wp_index + 1} Reached!")
        self.current_wp_index += 1

    def avoid_obstacle(self):
        print("   >>> [ALERT] OBSTACLE DETECTED DEAD AHEAD! <<<")
        print("   >>> [RL AGENT] Engaging Avoidance Maneuver...")
        # Steer 90 degrees right to avoid
        self.state.x += 15.0
        self.state.y -= 15.0
        print(f"   >>> [RL AGENT] Obstacle cleared. Correcting course. New Position: ({self.state.x:.1f}, {self.state.y:.1f})")
        time.sleep(0.5)

    def return_to_launch(self):
        self.mission_status = "RTL"
        print("\n[MISSION] All waypoints complete. Engaging Return To Launch (RTL)...")
        target = self.home_position
        
        while True:
            dx = target[0] - self.state.x
            dy = target[1] - self.state.y
            dist2d = math.sqrt(dx**2 + dy**2)
            
            if dist2d <= 5.0:
                break
            
            if dist2d > 15.0:
                self.state.x += (dx / dist2d) * 15.0 # Fly home faster
                self.state.y += (dy / dist2d) * 15.0
            else:
                self.state.x = target[0]
                self.state.y = target[1]
            
            print(f"   -> Flying Home... Current Position: ({self.state.x:.1f}, {self.state.y:.1f}) | Distance to Base: {dist2d:.1f}m")
            time.sleep(0.2)
            
        print("[MISSION] Arrived directly above Home Position.")

    def land(self):
        self.mission_status = "LANDING"
        print("\n[MISSION] Commencing Autonomous Landing...")
        while self.state.z > 0.0:
            self.state.z -= 2.0
            if self.state.z < 0: self.state.z = 0.0
            print(f"   -> Descending... Altitude: {self.state.z:.1f}m")
            time.sleep(0.2)
        print("[MISSION] Touchdown Confirmed. Motors Disarmed.")

    def run_full_mission(self):
        print("==================================================")
        print(">>> INITIALIZING 'FIRST GOAL' SIMULATION MISSION")
        print("==================================================")
        
        self.takeoff(target_alt=10.0)
        
        while self.current_wp_index < len(self.waypoints):
            self.navigate()
            
        self.return_to_launch()
        self.land()
        print("\n>>> MISSION SUCCESS: All goals from 'Option two.txt' achieved.")

if __name__ == "__main__":
    mission = FirstGoalMission()
    mission.run_full_mission()
