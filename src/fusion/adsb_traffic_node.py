import time
import math
import json
import random

class ADSBAwarenessNode:
    """
    AEGIS AUTONOMY: ADS-B Traffic Awareness Engine
    Listens to live 1090MHz transponder data from commercial aircraft 
    (via dump1090 or FlightAware dongle) and detects mid-air collision threats.
    """
    def __init__(self):
        # The drone's current position (Mocked for this test)
        self.drone_lat = 37.7749
        self.drone_lng = -122.4194
        self.drone_alt_m = 50.0
        
        # TCAS (Traffic Collision Avoidance System) parameters
        self.warning_radius_m = 2000.0 # 2km warning radius
        self.critical_radius_m = 500.0 # 500m evasive action radius
        
        print(">>> INITIALIZING ADS-B TRAFFIC AWARENESS NODE (1090 MHz)...")

    def haversine_distance(self, lat1, lon1, lat2, lon2):
        """Calculates distance in meters between two GPS coordinates."""
        R = 6371000
        phi_1, phi_2 = math.radians(lat1), math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)
        a = math.sin(delta_phi / 2.0)**2 + math.cos(phi_1) * math.cos(phi_2) * math.sin(delta_lambda / 2.0)**2
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    def fetch_live_adsb_data(self):
        """
        In production, this reads JSON from dump1090 (e.g., http://localhost:8080/data/aircraft.json).
        For this test, we simulate airplanes flying around San Francisco.
        """
        return [
            # A safe commercial jet high overhead
            {"hex": "A1B2C3", "flight": "UAL452", "lat": 37.7800, "lon": -122.4200, "alt_geom": 10000.0, "speed": 450},
            # A Cessna flying terrifyingly close to our drone
            {"hex": "D4E5F6", "flight": "N777XX", "lat": 37.7760, "lon": -122.4180, "alt_geom": 60.0, "speed": 110}
        ]

    def monitor_airspace(self):
        print("[ADS-B] Scanning local airspace for transponders...")
        aircraft_list = self.fetch_live_adsb_data()
        
        for aircraft in aircraft_list:
            dist_m = self.haversine_distance(self.drone_lat, self.drone_lng, aircraft["lat"], aircraft["lon"])
            alt_diff_m = abs(self.drone_alt_m - aircraft["alt_geom"])
            
            # If it's a commercial jet at 30,000 feet, we don't care
            if alt_diff_m > 300.0:
                print(f"[ADS-B] {aircraft['flight']} detected at {aircraft['alt_geom']}m (Distance: {dist_m/1000:.1f}km) - NO THREAT")
                continue
                
            # If it's in our altitude band, check the radius
            if dist_m < self.critical_radius_m:
                print(f"\n[TCAS] !!! CRITICAL TRAFFIC ALERT !!!")
                print(f"[TCAS] Aircraft {aircraft['flight']} is {dist_m:.1f}m away at our altitude!")
                print("[TCAS] Initiating emergency vertical descent to avoid mid-air collision!\n")
                
                # In ROS 2, this would publish a Twist message commanding an instant -Z velocity
                
            elif dist_m < self.warning_radius_m:
                print(f"[TCAS] WARNING: Traffic {aircraft['flight']} approaching. Distance: {dist_m:.1f}m")

if __name__ == "__main__":
    node = ADSBAwarenessNode()
    
    # Simulate a few radar sweeps
    for _ in range(3):
        node.monitor_airspace()
        time.sleep(1)
        # Simulate the Cessna getting closer...
        node.drone_lat += 0.0001
        node.drone_lng += 0.0001
