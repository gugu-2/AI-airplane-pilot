import json
import math

class GeofenceEngine:
    """
    AEGIS AUTONOMY: Aerospace Geofencing Engine
    Mathematically prevents the AI from flying into restricted airspace (Airports, Military bases, etc.)
    """
    def __init__(self):
        # In a real system, these would be loaded dynamically from an FAA/DGCA API.
        # Format: List of Polygons, where each Polygon is a list of (Lat, Lng) tuples.
        self.restricted_zones = [
            {
                "name": "San Francisco International Airport (SFO)",
                "polygon": [
                    (37.640, -122.408),
                    (37.640, -122.361),
                    (37.596, -122.361),
                    (37.596, -122.408)
                ]
            },
            {
                "name": "Area 51 Military Base",
                "polygon": [
                    (37.287, -115.864),
                    (37.287, -115.753),
                    (37.200, -115.753),
                    (37.200, -115.864)
                ]
            }
        ]
        
        # A hard limit on how far the drone can fly from its launch point
        self.max_radius_meters = 5000.0
        self.launch_lat = None
        self.launch_lng = None

    def set_launch_point(self, lat, lng):
        self.launch_lat = lat
        self.launch_lng = lng
        print(f"[GEOFENCE] Launch point locked at ({lat}, {lng}). Max radius: {self.max_radius_meters}m.")

    def haversine_distance(self, lat1, lon1, lat2, lon2):
        """Calculates distance in meters between two GPS coordinates."""
        R = 6371000  # Radius of Earth in meters
        phi_1, phi_2 = math.radians(lat1), math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)
        
        a = math.sin(delta_phi / 2.0)**2 + \
            math.cos(phi_1) * math.cos(phi_2) * \
            math.sin(delta_lambda / 2.0)**2
            
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c

    def is_point_in_polygon(self, lat, lng, polygon):
        """
        Uses the Ray-Casting algorithm to determine if a GPS coordinate is inside a polygon.
        """
        inside = False
        n = len(polygon)
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if min(p1y, p2y) < lng <= max(p1y, p2y):
                if lat <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (lng - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or lat <= xinters:
                        inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    def validate_waypoint(self, target_lat, target_lng):
        """
        The absolute safety firewall. If the AI requests a waypoint, it MUST pass this check.
        """
        # 1. Check Max Radius (Fly-away protection)
        if self.launch_lat is not None and self.launch_lng is not None:
            dist = self.haversine_distance(self.launch_lat, self.launch_lng, target_lat, target_lng)
            if dist > self.max_radius_meters:
                print(f"[GEOFENCE] [REJECTED] Waypoint is {dist:.1f}m away. Exceeds max radius of {self.max_radius_meters}m.")
                return False

        # 2. Check Restricted Airspace Polygons
        for zone in self.restricted_zones:
            if self.is_point_in_polygon(target_lat, target_lng, zone["polygon"]):
                print(f"[GEOFENCE] [REJECTED] Waypoint intersects restricted airspace -> {zone['name']}")
                return False

        print("[GEOFENCE] [OK] Waypoint Validated. Airspace clear.")
        return True

if __name__ == "__main__":
    # --- Quick Test of the Geofence Engine ---
    engine = GeofenceEngine()
    engine.set_launch_point(37.7749, -122.4194) # Launch from downtown SF
    
    print("\n--- Testing Safe Waypoint ---")
    engine.validate_waypoint(37.7750, -122.4190) # Just down the street
    
    print("\n--- Testing Restricted Airport Waypoint ---")
    engine.validate_waypoint(37.620, -122.380) # Right inside SFO Airport
    
    print("\n--- Testing Fly-Away Distance ---")
    engine.validate_waypoint(38.0, -122.0) # Too far away
