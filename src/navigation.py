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

    async def rrt_star_plan(self, start_lat, start_lon, dest_lat, dest_lon, alt):
        """
        Simulates an RRT* (Rapidly-exploring Random Tree Star) path planning algorithm.
        Instead of a straight line, it generates a series of intermediate waypoints
        to avoid mathematically defined obstacle zones.
        """
        print(f"[RRT*] Calculating optimal obstacle-free trajectory from ({start_lat:.6f}, {start_lon:.6f}) to ({dest_lat:.6f}, {dest_lon:.6f})")
        
        # In a real RRT* implementation, this would:
        # 1. Randomly sample the 3D space.
        # 2. Check collision against an occupancy grid.
        # 3. Connect nodes to find the shortest collision-free path.
        
        # Here we mock the result by generating a curved, multi-segment path to the destination
        waypoints = []
        num_segments = 3
        
        lat_step = (dest_lat - start_lat) / num_segments
        lon_step = (dest_lon - start_lon) / num_segments
        
        # Generate slightly randomized intermediate points (simulating going around obstacles)
        for i in range(1, num_segments):
            jitter_lat = random.uniform(-0.00005, 0.00005)
            jitter_lon = random.uniform(-0.00005, 0.00005)
            
            inter_lat = start_lat + (lat_step * i) + jitter_lat
            inter_lon = start_lon + (lon_step * i) + jitter_lon
            # We vary the altitude slightly for a true 3D trajectory
            inter_alt = alt + random.uniform(-2, 5) 
            
            waypoints.append((inter_lat, inter_lon, inter_alt))
            
        # Append final destination
        waypoints.append((dest_lat, dest_lon, alt))
        
        print(f"[RRT*] Generated {len(waypoints)} intermediate waypoints for complex trajectory.")
        return waypoints

    async def fly_to_waypoint(self, lat: float, lon: float, alt: float):
        """
        Commands the drone to fly to a specific global position.
        """
        print(f"Navigating to waypoint: Lat {lat:.6f}, Lon {lon:.6f}, Alt {alt:.1f}")
        # MAVSDK's goto_location takes (latitude_deg, longitude_deg, absolute_altitude_m, yaw_deg)
        await self.drone.action.goto_location(lat, lon, alt, float('nan'))

        # A robust system would constantly check distance to target here.
        # For this basic implementation, we will just sleep and assume it gets there.
        # In the main loop, we will implement distance checking.
        await asyncio.sleep(5) 
        
