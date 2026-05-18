import random
import time
import math

class ADSBSensor:
    """
    Simulates a connection to an ADS-B Receiver or the OpenSky Network API.
    Provides live coordinates of nearby commercial and private aircraft.
    """
    def __init__(self, anchor_lat, anchor_lon):
        self.anchor_lat = anchor_lat
        self.anchor_lon = anchor_lon
        
        # Generate some mock flights
        self.aircraft = [
            {"callsign": "UAL128", "lat": anchor_lat + 0.05, "lon": anchor_lon - 0.05, "alt": 1500.0, "heading": 135, "speed": 150},
            {"callsign": "SWA442", "lat": anchor_lat - 0.03, "lon": anchor_lon + 0.04, "alt": 2200.0, "heading": 310, "speed": 180},
            {"callsign": "N12345", "lat": anchor_lat + 0.01, "lon": anchor_lon + 0.01, "alt": 600.0, "heading": 180, "speed": 60}
        ]
        self.last_update = time.time()

    def get_live_traffic(self):
        """
        Calculates new positions based on heading and speed.
        Returns a list of aircraft dictionaries.
        """
        current_time = time.time()
        dt = current_time - self.last_update
        self.last_update = current_time
        
        for ac in self.aircraft:
            # Roughly: 1 degree lat/lon ≈ 111,320 meters
            speed_deg = ac["speed"] / 111320.0
            rad = math.radians(ac["heading"])
            ac["lat"] += (speed_deg * math.cos(rad)) * dt
            ac["lon"] += (speed_deg * math.sin(rad)) * dt
            
        return self.aircraft

    def check_tcas_alarm(self, drone_lat, drone_lon, drone_alt):
        """
        Traffic Collision Avoidance System (TCAS).
        Returns True if an aircraft is dangerously close.
        """
        for ac in self.aircraft:
            R = 6371000
            phi1, phi2 = math.radians(drone_lat), math.radians(ac["lat"])
            dphi = math.radians(ac["lat"] - drone_lat)
            dlambda = math.radians(ac["lon"] - drone_lon)
            a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
            dist_m = R * c
            
            alt_diff = abs(drone_alt - ac["alt"])
            
            # If within 1500m horizontally and 200m vertically
            if dist_m < 1500.0 and alt_diff < 200.0:
                return True, ac["callsign"]
                
        return False, None
