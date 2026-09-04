import time
import threading
import random

# ==========================================================
# MOCK ROS 2 PUB/SUB BROKER (For Windows Demonstration)
# ==========================================================
class MockROS2Broker:
    def __init__(self):
        self.topics = {
            '/sensors/gps/raw': None,
            '/sensors/imu/raw': None,
            '/aegis/planner/target_waypoint': None,
            '/hardware/actuators/cmd_raw': None,
            '/hardware/actuators/cmd_safe': None,
            '/aegis/mission/status': None,
            '/communications/vhf_radio/rx': None,
            '/aegis/mission/atc_override': None,
            '/aegis/perception/identified_objects': None,
            '/hardware/human_pilot/yoke': None,
            '/mavros/state': None,
            '/camera/rgb/image_raw': None
        }
        
broker = MockROS2Broker()

# ==========================================================
# LAYER 1: SENSORS & HARDWARE INTERFACE
# ==========================================================
def hardware_sensor_node():
    print("[Node Started] Hardware Interface (Layer 1)")
    while True:
        # Navigation (GNSS/GPS, IMU)
        broker.topics['/sensors/gps/raw'] = {'x': random.uniform(0, 100), 'y': 0, 'z': 500}
        broker.topics['/sensors/imu/raw'] = {'accel_x': random.uniform(-1, 1)}
        
        # Vision (EO/IR) & Spatial (LiDAR) & Avionics (Pitot)
        if random.random() < 0.2:
            print("[Hardware Node] [VISION] IR Camera Feed processed.")
            print("[Hardware Node] [SPATIAL] LiDAR pointcloud terrain mapped.")
            print("[Hardware Node] [AVIONICS] Pitot Tube Airspeed: 125 knots.")
            
        time.sleep(1.0) # 1Hz GPS

# ==========================================================
# LAYER 2: SENSOR FUSION (EKF)
# ==========================================================
def sensor_fusion_node():
    print("[Node Started] Extended Kalman Filter (Layer 2)")
    while True:
        gps = broker.topics['/sensors/gps/raw']
        imu = broker.topics['/sensors/imu/raw']
        if gps and imu:
            # Simulate Kalman Fusion
            fused_x = gps['x'] * 0.8 + (imu['accel_x'] * 0.2)
            broker.topics['/aegis/state/fused_position'] = {'x': fused_x, 'y': gps['y'], 'z': gps['z']}
            print(f"[EKF Node] Fused GPS and IMU -> Published Clean State: X={fused_x:.2f}")
        time.sleep(0.5)

# ==========================================================
# LAYER 3: COGNITIVE PLANNER (PyTorch RL)
# ==========================================================
def cognitive_planner_node():
    print("[Node Started] PyTorch RL Planner (Layer 3)")
    while True:
        state = broker.topics['/aegis/state/fused_position']
        if state:
            # Simulate Neural Network Inference
            target_z = state['z'] + random.choice([-10.0, 0.0, 10.0])
            action_name = "CLIMB" if target_z > state['z'] else "DESCEND" if target_z < state['z'] else "MAINTAIN"
            
            broker.topics['/aegis/planner/target_waypoint'] = {'x': state['x'] + 10, 'y': state['y'], 'z': target_z}
            print(f"[RL Node] State Received. Inference: {action_name} -> Published Target: Z={target_z:.1f}")
        time.sleep(1.0)

# ==========================================================
# LAYER 4: FLIGHT CONTROL (4-Axis Fly-by-Wire)
# ==========================================================
def flight_control_node():
    print("[Node Started] Fly-by-Wire PID Controller (Layer 4)")
    while True:
        target = broker.topics['/aegis/planner/target_waypoint']
        state = broker.topics['/aegis/state/fused_position']
        if target and state:
            # 4 Independent PID Loops
            elevator_cmd = (target['z'] - state['z']) * 1.5
            aileron_cmd = (target['y'] - state['y']) * 1.2
            rudder_cmd = aileron_cmd * 0.3
            throttle_cmd = 75.0
            
            broker.topics['/hardware/actuators/cmd_raw'] = {
                'elevator': elevator_cmd, 'aileron': aileron_cmd, 
                'rudder': rudder_cmd, 'throttle': throttle_cmd
            }
            print(f"[FBW Node] AI commands 4-Axes -> Elev: {elevator_cmd:.1f} | Ail: {aileron_cmd:.1f} | Rud: {rudder_cmd:.1f} | Thr: {throttle_cmd:.1f}%")
        time.sleep(0.5)

# ==========================================================
# SAFETY: ENVELOPE PROTECTION
# ==========================================================
def envelope_protection_node():
    print("[Node Started] Envelope Protection Watchdog (Safety Firewall)")
    MAX_PITCH = 15.0
    while True:
        raw_cmd = broker.topics.get('/hardware/actuators/cmd_raw')
        if raw_cmd:
            safe_elev = raw_cmd['elevator']
            
            # The AI can occasionally output insane values if it hallucinates.
            # We inject a simulated AI glitch here to prove the firewall works.
            if random.random() < 0.1:
                safe_elev = 45.0 # AI hallucinates a violent 45-degree pitch up
                print(f"[AI GLITCH] The Neural Network just requested a {safe_elev} degree pitch!")
                
            if safe_elev > MAX_PITCH:
                print(f"[ENVELOPE PROTECTION] BLOCKED! Requested pitch {safe_elev:.1f} > MAX {MAX_PITCH}. Clamping to 15.0!")
                safe_elev = MAX_PITCH
            elif safe_elev < -MAX_PITCH:
                print(f"[ENVELOPE PROTECTION] BLOCKED! Requested pitch {safe_elev:.1f} < MIN -{MAX_PITCH}. Clamping to -15.0!")
                safe_elev = -MAX_PITCH
                
            broker.topics['/hardware/actuators/cmd_safe'] = {
                'elevator': safe_elev, 'aileron': raw_cmd['aileron'], 
                'rudder': raw_cmd['rudder'], 'throttle': raw_cmd['throttle']
            }
        time.sleep(0.1)

# ==========================================================
# LAYER 3: COGNITIVE (NLP & CNN)
# ==========================================================
def atc_nlp_node():
    print("[Node Started] ATC NLP Agent (Layer 3)")
    transcripts = [
        "Aegis 1, cleared to land.",
        "Tower to Aegis 1, abort landing, go around!",
        "Delta 452, maintain heading."
    ]
    while True:
        if random.random() < 0.15: # Randomly receive ATC message
            msg = random.choice(transcripts)
            broker.topics['/communications/vhf_radio/rx'] = msg
            print(f"\n[ATC Node] VHF RX: \"{msg}\"")
            if "Aegis 1" in msg and "abort" in msg:
                print(f"[ATC Node] INTENT: ABORT_LANDING. Transmitting: \"Aegis 1, going around.\"")
                broker.topics['/aegis/mission/atc_override'] = "ABORT_LANDING"
        time.sleep(1.0)

def computer_vision_node():
    print("[Node Started] CNN Computer Vision (Layer 3)")
    while True:
        if random.random() < 0.1: # Spot traffic
            print("[CNN Node] Airborne Traffic detected! Publishing to avoidance planner.")
            broker.topics['/aegis/perception/identified_objects'] = {'class': 'aircraft'}
        time.sleep(0.5)

# ==========================================================
# LAYER 6: MISSION MANAGEMENT
# ==========================================================
def mission_management_node():
    print("[Node Started] Mission Management (Layer 6)")
    status = "BOARDING"
    pax = 4
    pressure = 14.7
    
    while True:
        if status == "BOARDING":
            print(f"[Mission Node] Flight AEGIS-77: {status} ({pax} pax). Departure in 10s.")
            time.sleep(2)
            status = "IN_FLIGHT"
        else:
            pressure += random.uniform(-0.1, 0.1)
            print(f"[Mission Node] Airspace Nominal. Geofence OK. Cabin Pressure: {pressure:.1f} PSI.")
            time.sleep(3)

# ==========================================================
# PHASE 3: SHADOW MODE EVALUATOR
# ==========================================================
def shadow_mode_node():
    print("[Node Started] Shadow Mode Evaluator (Phase 3)")
    errors = 0
    total = 0
    while True:
        # Mock human pilot inputs
        human_elev = random.choice([0.0, 0.0, 10.0, -10.0]) # Human mostly flies straight
        broker.topics['/hardware/human_pilot/yoke'] = {'elevator': human_elev}
        
        ai_cmd = broker.topics.get('/hardware/actuators/cmd_safe')
        if ai_cmd:
            total += 1
            ai_elev = ai_cmd['elevator']
            if abs(ai_elev - human_elev) > 10.0:
                errors += 1
                print(f"[SHADOW MODE WARNING] AI wanted {ai_elev:.1f} but Human did {human_elev:.1f}. Error Rate: {(errors/total)*100:.1f}%")
            else:
                # To avoid spamming, only print consensus occasionally
                if random.random() < 0.2:
                    print(f"[SHADOW MODE] AI in consensus with Human Pilot.")
        time.sleep(0.5)

# ==========================================================
# PHASE 2: MAVROS HOOP MISSION (CV Integration)
# ==========================================================
def mavros_hoop_mission_node():
    print("[Node Started] MAVROS Hoop Mission (Phase 2 CV Demo)")
    state = "INIT"
    hoop_distance = 100.0
    
    while True:
        if state == "INIT":
            print("[MAVROS] Arming motors and taking off to 3.0m...")
            time.sleep(2)
            state = "SEARCHING_HOOP"
            
        elif state == "SEARCHING_HOOP":
            print("[MAVROS] OpenCV: Scanning camera feed for Red Hoop...")
            if random.random() < 0.3: # 30% chance to find it each tick
                print("[MAVROS] OpenCV: Hoop detected at pixel (315, 242)!")
                state = "FLYING_THROUGH_HOOP"
                
        elif state == "FLYING_THROUGH_HOOP":
            if hoop_distance > 0:
                print(f"[MAVROS] Visual Servoing: Distance to Hoop: {hoop_distance:.1f}m. Adjusting Yaw/Pitch...")
                hoop_distance -= 25.0
            else:
                print("[MAVROS] SUCCESS! Flew through the hoop center.")
                state = "LANDING"
                
        elif state == "LANDING":
            print("[MAVROS] Initiating autonomous landing sequence.")
            time.sleep(2)
            print("[MAVROS] Touchdown confirmed. Mission Complete.")
            state = "DONE"
            
        elif state == "DONE":
            pass # Keep alive without spamming
            
        time.sleep(1.0)

# ==========================================================
# PERCEPTION & NAVIGATION: OPTICAL FLOW / SLAM
# ==========================================================
def visual_navigation_node():
    print("[Node Started] Visual SLAM & Optical Flow (Layers 3/4)")
    while True:
        if random.random() < 0.2:
            drift_x = random.uniform(-0.1, 0.1)
            print(f"[OPTICAL FLOW] Ground pixel tracking active. Countering drift: {drift_x:.2f} m/s")
            print(f"[VISUAL SLAM] 3D Depth Map updated. Odometry locked without GPS.")
        time.sleep(1.0)

# ==========================================================
# PHASE 4 | MODULE 4: EMERGENCY AI
# ==========================================================
def emergency_ai_node():
    print("[Node Started] Emergency AI Monitor (Module 4)")
    while True:
        r = random.random()
        if r < 0.05:
            print("\n[EMERGENCY AI] FATAL: GPS SIGNAL LOST (Spoofing Detected).")
            print("[EMERGENCY AI] Engaging Visual Odometry SLAM Fallback.\n")
        elif r > 0.95:
            print("\n[EMERGENCY AI] SEVERE WIND TURBULENCE DETECTED.")
            print("[EMERGENCY AI] Dynamically adjusting PID Derivative (Kd) gains to prevent stall.\n")
        time.sleep(2.0)

if __name__ == "__main__":
    print("==================================================")
    print(">>> AEGIS AUTONOMOUS OS: LAUNCHING ROS 2 NODES")
    print("==================================================\n")
    
    threads = [
        threading.Thread(target=hardware_sensor_node, daemon=True),
        threading.Thread(target=sensor_fusion_node, daemon=True),
        threading.Thread(target=cognitive_planner_node, daemon=True),
        threading.Thread(target=atc_nlp_node, daemon=True),
        threading.Thread(target=computer_vision_node, daemon=True),
        threading.Thread(target=flight_control_node, daemon=True),
        threading.Thread(target=envelope_protection_node, daemon=True),
        threading.Thread(target=mission_management_node, daemon=True),
        threading.Thread(target=shadow_mode_node, daemon=True),
        threading.Thread(target=mavros_hoop_mission_node, daemon=True),
        threading.Thread(target=visual_navigation_node, daemon=True),
        threading.Thread(target=emergency_ai_node, daemon=True)
    ]
    
    for t in threads:
        t.start()
        time.sleep(0.2)
        
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down Aegis OS...")
