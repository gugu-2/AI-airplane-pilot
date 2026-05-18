import sys
import os
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/control')))
from pid_bridge import CppPIDController

def simulate_step_response(kp, ki, kd, setpoint=10.0, steps=50):
    """
    Simulates how the aircraft pitch responds to a sudden command (e.g., pitch to 10 degrees).
    Returns the time history and pitch history.
    """
    # Initialize our C++ PID Bridge
    pid = CppPIDController(kp, ki, kd, min_out=-45.0, max_out=45.0)
    
    current_pitch = 0.0
    history = []
    
    # Simple aircraft physics mock
    # Actuator command changes pitch rate, which changes pitch.
    pitch_rate = 0.0 
    dt = 0.1
    
    for _ in range(steps):
        history.append(current_pitch)
        
        # 1. PID computes the required elevator angle
        elevator_cmd = pid.compute(setpoint, current_pitch, dt)
        
        # 2. Physics simulation: elevator deflects air, creating a pitch rate
        pitch_rate += (elevator_cmd * 0.1) * dt # Simplified aerodynamics
        
        # Add some aerodynamic damping (drag)
        pitch_rate *= 0.9
        
        # 3. Update current pitch
        current_pitch += pitch_rate * dt

    return history

if __name__ == "__main__":
    print("==================================================")
    print("AEROSPACE ENGINEERING: C++ PID GAIN TUNING")
    print("==================================================")
    
    # We will test 3 different tuning configurations
    configs = [
        {"name": "Aggressive (High Kp)", "kp": 3.0, "ki": 0.0, "kd": 0.1},
        {"name": "Sluggish (Low Kp)", "kp": 0.5, "ki": 0.1, "kd": 0.5},
        {"name": "Optimal (Tuned PID)", "kp": 1.2, "ki": 0.05, "kd": 1.5}
    ]
    
    for config in configs:
        history = simulate_step_response(config["kp"], config["ki"], config["kd"])
        
        # Print a simple text-based graph for the terminal
        print(f"\n--- Testing: {config['name']} (Kp={config['kp']}, Ki={config['ki']}, Kd={config['kd']}) ---")
        print("Target Setpoint: 10.0 degrees")
        
        for i, pitch in enumerate(history):
            if i % 5 == 0: # Print every 5th step to save space
                # Create a text bar chart
                bar_length = int(max(0, pitch) * 3)
                bar = "=" * bar_length
                print(f"Time {i*0.1:4.1f}s | Pitch: {pitch:5.2f} | {bar}")
                
        # Calculate performance metrics
        overshoot = max(0, max(history) - 10.0)
        final_error = abs(10.0 - history[-1])
        print(f"-> Max Overshoot: {overshoot:.2f} degrees")
        print(f"-> Steady State Error: {final_error:.2f} degrees")
