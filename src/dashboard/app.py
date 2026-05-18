from flask import Flask, render_template
from flask_socketio import SocketIO
import threading
import time
import math
import random

app = Flask(__name__)
app.config['SECRET_KEY'] = 'aegis-secret!'
socketio = SocketIO(app, cors_allowed_origins="*")

# Simulated Drone State (This would normally read from ROS 2 / MAVSDK)
drone_state = {
    'lat': 37.7749, # San Francisco origin
    'lng': -122.4194,
    'alt': 0.0,
    'heading': 0.0,
    'battery': 100.0,
    'status': 'STANDBY'
}

@app.route('/')
def index():
    return render_template('index.html')

def telemetry_loop():
    """Simulates real-time telemetry from the drone"""
    global drone_state
    
    # Simulate a basic flight path for the dashboard
    time.sleep(5)
    drone_state['status'] = 'TAKEOFF'
    for _ in range(10):
        drone_state['alt'] += 1.0
        socketio.emit('telemetry', drone_state)
        time.sleep(0.5)
        
    drone_state['status'] = 'NAVIGATING'
    for _ in range(200):
        # Fly roughly north-east
        drone_state['lat'] += 0.00005
        drone_state['lng'] += 0.00003
        drone_state['heading'] = 30.0
        drone_state['battery'] -= 0.1
        
        # Simulate slight wind turbulence
        drone_state['alt'] += random.uniform(-0.2, 0.2)
        
        socketio.emit('telemetry', drone_state)
        time.sleep(0.5)

if __name__ == '__main__':
    print(">>> STARTING AEGIS GROUND CONTROL STATION (GCS) ON PORT 5000")
    
    # Start the telemetry simulation thread
    t = threading.Thread(target=telemetry_loop, daemon=True)
    t.start()
    
    # Run the Flask WebSockets Server
    socketio.run(app, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)
