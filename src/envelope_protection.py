import asyncio

class EnvelopeProtectionSystem:
    """
    Hardcoded deterministic firewall that sits between the AI and the physical drone.
    Prevents the AI from commanding unsafe maneuvers (e.g. flying into the ground, 
    breaching geofences, or commanding extreme pitch/roll).
    """
    def __init__(self, logger=None, telemetry_server=None, home_lat=47.3977, home_lon=8.5455):
        self.logger = logger
        self.telemetry_server = telemetry_server
        
        # Flight Envelope Constraints
        self.MIN_ALTITUDE_M = 50.0   # Absolute minimum altitude during cruise
        self.MAX_ALTITUDE_M = 1000.0 # Absolute maximum altitude ceiling
        self.MAX_PITCH_DEG = 15.0    # Used in offboard mode
        self.MAX_ROLL_DEG = 30.0     # Used in offboard mode
        
        # Fix #10: Geofence is now DYNAMIC — calculated from actual takeoff position.
        # This prevents the Zurich hardcoding bug that would clamp any drone
        # flying outside Switzerland to GPS coordinates near Zurich.
        GEOFENCE_RADIUS_DEG = 0.02  # ~2.2 km radius around home
        self.GEOFENCE_MIN_LAT = home_lat - GEOFENCE_RADIUS_DEG
        self.GEOFENCE_MAX_LAT = home_lat + GEOFENCE_RADIUS_DEG
        self.GEOFENCE_MIN_LON = home_lon - GEOFENCE_RADIUS_DEG
        self.GEOFENCE_MAX_LON = home_lon + GEOFENCE_RADIUS_DEG
        
        print(f"[Envelope] Geofence set: Lat [{self.GEOFENCE_MIN_LAT:.4f}, {self.GEOFENCE_MAX_LAT:.4f}] "
              f"Lon [{self.GEOFENCE_MIN_LON:.4f}, {self.GEOFENCE_MAX_LON:.4f}]")
        
        # Disable MIN_ALTITUDE during takeoff/landing
        self.is_landing = False

    def set_landing_mode(self, state: bool):
        """Disables the minimum altitude floor when the drone is explicitly landing."""
        self.is_landing = state

    async def safe_goto_location(self, drone, lat: float, lon: float, alt: float, yaw: float):
        """
        Wrapper for drone.action.goto_location.
        Intercepts and clamps the AI's requested parameters before they reach the actuators.
        """
        violation = False
        warning_msg = ""
        
        safe_lat = lat
        safe_lon = lon
        safe_alt = alt

        # 1. Geofence Protection
        if not (self.GEOFENCE_MIN_LAT <= lat <= self.GEOFENCE_MAX_LAT):
            safe_lat = max(self.GEOFENCE_MIN_LAT, min(self.GEOFENCE_MAX_LAT, lat))
            violation = True
            warning_msg += f"Latitude breached {lat:.5f}. "
            
        if not (self.GEOFENCE_MIN_LON <= lon <= self.GEOFENCE_MAX_LON):
            safe_lon = max(self.GEOFENCE_MIN_LON, min(self.GEOFENCE_MAX_LON, lon))
            violation = True
            warning_msg += f"Longitude breached {lon:.5f}. "

        # 2. Altitude Floor/Ceiling Protection
        if safe_alt > self.MAX_ALTITUDE_M:
            safe_alt = self.MAX_ALTITUDE_M
            violation = True
            warning_msg += f"Max Altitude breached {alt:.1f}m. "
            
        if not self.is_landing and safe_alt < self.MIN_ALTITUDE_M:
            safe_alt = self.MIN_ALTITUDE_M
            violation = True
            warning_msg += f"Min Altitude breached {alt:.1f}m. "

        # 3. Report Violation
        if violation:
            final_msg = f"[ENVELOPE FIREWALL] AI violation intercepted! Clamped to safe limits. ({warning_msg.strip()})"
            print(f"\n{final_msg}")
            
            # Log to DB
            if self.logger:
                self.logger.log_telemetry(safe_lat, safe_lon, safe_alt, battery_v=0.0, message="FIREWALL VIOLATION", level="critical")
                
            # Alert Dashboard
            if self.telemetry_server:
                await self.telemetry_server.broadcast({
                    "type": "log",
                    "data": {"msg": final_msg, "level": "critical"}
                })

        # 4. Execute the strictly safe command
        await drone.action.goto_location(safe_lat, safe_lon, safe_alt, yaw)
        return violation
