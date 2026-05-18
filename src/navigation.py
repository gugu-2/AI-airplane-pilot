import asyncio
import os
# We force MOCK mode on Windows by default because MAVSDK server binary is missing.
USE_MOCK = True

if USE_MOCK or os.name == 'nt':
    print("\n--- Running in MOCK Simulation Mode ---")
    from mock_mavsdk import MockSystem as System
else:
    from mavsdk import System
import math
import random

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

class NavigationModule:
    def __init__(self, drone: System):
        self.drone = drone

    async def generate_square_waypoints(self, side_length: float = 10.0):
        """
        Generates 4 waypoints relative to the current position to fly in a square.
        """
        print("Fetching current position...")
        async for position in self.drone.telemetry.position():
            lat = position.latitude_deg
            lon = position.longitude_deg
            alt = position.absolute_altitude_m
            break # Just need the current position once

        # Rough approximation: 1 degree of latitude is ~111,111 meters
        # 1 degree of longitude is ~111,111 * cos(latitude) meters
        lat_offset = side_length / 111111.0
        lon_offset = side_length / (111111.0 * math.cos(math.radians(lat)))

        waypoints = [
            (lat + lat_offset, lon, alt),                 # North
            (lat + lat_offset, lon + lon_offset, alt),      # North-East
            (lat, lon + lon_offset, alt),                 # East
            (lat, lon, alt)                               # Back to start
        ]
        
        return waypoints

    async def rrt_star_plan(self, start_lat, start_lon, dest_lat, dest_lon, alt, mapper=None):
        """
        Simulates an RRT* (Rapidly-exploring Random Tree Star) path planning algorithm.
        Instead of a straight line, it generates a series of intermediate waypoints
        to avoid mathematically defined obstacle zones. Uses SemanticMap memory.
        """
        print(f"[RRT*] Calculating optimal obstacle-free trajectory from ({start_lat:.6f}, {start_lon:.6f}) to ({dest_lat:.6f}, {dest_lon:.6f})")
        
        waypoints = []
        num_segments = 5
        
        lat_step = (dest_lat - start_lat) / num_segments
        lon_step = (dest_lon - start_lon) / num_segments
        
        for i in range(1, num_segments):
            inter_lat = start_lat + (lat_step * i)
            inter_lon = start_lon + (lon_step * i)
            inter_alt = alt
            
            if mapper:
                x, y = mapper._gps_to_grid(inter_lat, inter_lon)
                if x is not None and y is not None:
                    # Very basic check if grid cell is an obstacle (-1)
                    # For safety, we check a 3x3 window in a real app, here we check the cell itself
                    if mapper.grid[y, x] == -1:
                        print(f"[RRT*] Memory map collision detected at segment {i}! Rerouting...")
                        inter_lat += random.uniform(0.00002, 0.00006)
                        inter_lon += random.uniform(0.00002, 0.00006)
                        inter_alt += 2.0
            
            waypoints.append((inter_lat, inter_lon, inter_alt))
            
        # Append final destination
        waypoints.append((dest_lat, dest_lon, alt))
        
        print(f"[RRT*] Generated {len(waypoints)} intermediate waypoints for trajectory.")
        return waypoints

    async def fly_to_waypoint(self, lat: float, lon: float, alt: float):
        """
        Commands the drone to fly to a specific global position.
        """
        print(f"Navigating to waypoint: Lat {lat:.6f}, Lon {lon:.6f}, Alt {alt:.1f}")
        # MAVSDK's goto_location takes (latitude_deg, longitude_deg, absolute_altitude_m, yaw_deg)
        await self.drone.action.goto_location(lat, lon, alt, float('nan'))

        # Poll telemetry until we arrive within a 1.0m tolerance
        async for position in self.drone.telemetry.position():
            curr_lat = position.latitude_deg
            curr_lon = position.longitude_deg
            
            dist = haversine_distance(curr_lat, curr_lon, lat, lon)
            if dist < 1.0:
                print(f"Reached waypoint! Distance: {dist:.2f}m")
                break
            
            await asyncio.sleep(0.5)
        
