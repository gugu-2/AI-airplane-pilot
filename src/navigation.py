# Navigation Module — Aegis Autonomous Flight OS
# NOTE: This module does NOT import MAVSDK directly.
# The 'drone' system object is injected by main_pilot.py (Dependency Injection pattern),
# keeping this module fully testable without a live flight controller.
import asyncio
import math
import random
import sys
import os

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculates great-circle distance (meters) between two GPS coordinates."""
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

# Ensure cognitive module is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from cognitive.rl_models import RLInferenceEngine

class NavigationModule:
    def __init__(self, drone: System, envelope=None):
        self.drone = drone
        self.envelope = envelope
        self.rl_engine = None

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

    async def linear_waypoint_plan(self, start_lat, start_lon, dest_lat, dest_lon, alt, mapper=None):
        """
        Fix #6: Renamed from rrt_star_plan. This is honest linear interpolation, NOT a true RRT*.
        A real RRT* would use random tree sampling and rewiring (see docs for Gazebo integration).
        Generates intermediate waypoints between start and destination, with basic obstacle map checks.
        """
        print(f"[Waypoint] Calculating linear trajectory from ({start_lat:.6f}, {start_lon:.6f}) to ({dest_lat:.6f}, {dest_lon:.6f})")
        
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
        
        print(f"[Waypoint] Generated {len(waypoints)} intermediate waypoints for trajectory.")
        return waypoints

    async def cognitive_rl_plan(self, start_lat, start_lon, dest_lat, dest_lon, alt, mapper=None):
        """
        Fix #26: Uses asyncio.wait_for() to prevent PyTorch inference from blocking the event loop.
        Uses the PyTorch Deep Q-Network to dynamically generate a 3D path.
        """
        print(f"[RL-DQN] Neural Network taking over navigation to ({dest_lat:.6f}, {dest_lon:.6f})")
        
        if not self.rl_engine:
            self.rl_engine = RLInferenceEngine()
            
        async def _run_plan():
            loop = asyncio.get_event_loop()
            waypoints = []
            current_lat = start_lat
            current_lon = start_lon
            current_alt = alt
            max_steps = 20
            step_count = 0
            
            while step_count < max_steps:
                dist = haversine_distance(current_lat, current_lon, dest_lat, dest_lon)
                if dist < 5.0:
                    break
                    
                obs_dist = 100.0
                if mapper and len(mapper.obstacles) > 0:
                    obs_dist = min([
                        haversine_distance(current_lat, current_lon, obs['lat'], obs['lon'])
                        for obs in mapper.obstacles
                    ])
                
                # Run PyTorch inference in thread pool so it doesn't block the event loop (Fix #26)
                action_result = await loop.run_in_executor(
                    None,
                    self.rl_engine.decide_next_action,
                    current_lat, current_lon, dest_lat, dest_lon, obs_dist
                )
                action_name, d_lat, d_lon, d_alt = action_result
                print(f"[RL-DQN] Step {step_count+1}: Model chose action '{action_name}'")
                
                current_lat += d_lat
                current_lon += d_lon
                current_alt += d_alt
                waypoints.append((current_lat, current_lon, current_alt))
                step_count += 1
                
            waypoints.append((dest_lat, dest_lon, alt))
            return waypoints
        
        try:
            # 2-second timeout prevents blocking the async loop during cold GPU start
            return await asyncio.wait_for(_run_plan(), timeout=2.0)
        except asyncio.TimeoutError:
            print("[RL-DQN] WARNING: Inference timeout. Falling back to linear waypoint.")
            return await self.linear_waypoint_plan(start_lat, start_lon, dest_lat, dest_lon, alt, mapper)

    async def fly_to_waypoint(self, lat: float, lon: float, alt: float):
        """
        Commands the drone to fly to a specific global position.
        """
        print(f"Navigating to waypoint: Lat {lat:.6f}, Lon {lon:.6f}, Alt {alt:.1f}")
        # MAVSDK's goto_location takes (latitude_deg, longitude_deg, absolute_altitude_m, yaw_deg)
        if self.envelope:
            await self.envelope.safe_goto_location(self.drone, lat, lon, alt, float('nan'))
        else:
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
        
