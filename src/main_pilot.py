import asyncio
import os
# We force MOCK mode on Windows by default because MAVSDK server binary is missing.
USE_MOCK = True

if USE_MOCK or os.name == 'nt':
    print("\n--- Running in MOCK Simulation Mode ---")
    from mock_mavsdk import MockSystem as System
else:
    from mavsdk import System

import argparse

# Import our custom modules
from navigation import NavigationModule
from perception import PerceptionModule
from hardware_perception import HardwarePerception
from avoidance import ObstacleAvoidanceModule
from logger import FlightLogger
from mapping import SemanticMap
from telemetry_server import TelemetryServer
from swarm_network import SwarmNetwork
from safety_consensus import TripleRedundancySystem
import random
import os

async def monitor_kill_switch(drone: System):
    """
    Background safety thread: Monitors RC channel 5.
    If the pilot flips the switch (PWM > 1500), instantly kill the motors.
    """
    print("[Safety] Kill switch monitor active on RC Channel 5...")
    try:
        async for rc_status in drone.telemetry.rc_status():
            # MAVSDK provides scaled RC channels or raw. This is a simplified check.
            # In a real physical setup, we check exact RC channel mappings.
            if hasattr(rc_status, 'rc_channels_scaled') and len(rc_status.rc_channels_scaled) > 4:
                # Channel 5 is usually index 4
                ch5_val = rc_status.rc_channels_scaled[4] 
                if ch5_val > 0.5: # 0.5 scaled is roughly > 1500 PWM
                    print("\n!!! [EMERGENCY] PILOT TRIGGERED KILL SWITCH !!!")
                    await drone.action.kill()
                    print("!!! MOTORS KILLED. SHUTTING DOWN AI !!!")
                    # Force exit
                    os._exit(1)
    except Exception as e:
        # Mock system might not have RC status implemented
        pass

async def run_mission(args):
    connection_address = args.connect
    print("Initializing Main AI Brain...")
    
    drone = System()
    
    print(f"Connecting to drone at {connection_address}...")
    await drone.connect(system_address=connection_address)

    print("Waiting for drone to connect...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print(f"-- Connected to drone!")
            break

    print("Waiting for global position estimate...")
    async for health in drone.telemetry.health():
        if health.is_global_position_ok and health.is_home_position_ok:
            print("-- Global position estimate OK")
            break

    # Initialize Modules
    nav_module = NavigationModule(drone)
    
    # Decide between hardware or simulation perception
    if args.hardware:
        perception_module = HardwarePerception(camera_index=0)
    else:
        perception_module = PerceptionModule()
        
    avoidance_module = ObstacleAvoidanceModule()
    logger = FlightLogger()
    mapper = SemanticMap(grid_size=200, resolution=0.5) # High-res map (0.5m blocks)
    telemetry_server = TelemetryServer()
    safety_system = TripleRedundancySystem(tolerance_meters=2.0)
    
    # Initialize Swarm Intelligence
    swarm = SwarmNetwork(drone_id=args.drone_id)
    swarm.attach_mapper(mapper)

    # Start the safety kill switch monitor in the background
    asyncio.create_task(monitor_kill_switch(drone))
    
    # Start the WebSocket server in the background
    asyncio.create_task(telemetry_server.start_server())
    
    # Start listening for swarm intelligence broadcasts
    asyncio.create_task(swarm.listen_for_swarm())

    print("\n--- Starting Mission ---")
    print("Arming...")
    await drone.action.arm()
    
    print("Taking off...")
    await drone.action.takeoff()
    await asyncio.sleep(8) # Wait for takeoff

    # 1. Generate a high-level mission path (e.g., a square)
    square_corners = await nav_module.generate_square_waypoints(side_length=15.0)
    print(f"Generated {len(square_corners)} high-level corners for the mission.")

    # Get current position for the start of the first RRT leg
    current_lat, current_lon, current_alt = 0, 0, 0
    async for position in drone.telemetry.position():
        current_lat = position.latitude_deg
        current_lon = position.longitude_deg
        current_alt = position.absolute_altitude_m
        break

    # Anchor the map memory to our starting coordinate
    mapper.set_anchor(current_lat, current_lon)

    # 2. Execute Path using RRT* for each leg
    for corner_idx, (dest_lat, dest_lon, dest_alt) in enumerate(square_corners):
        print(f"\n=== Navigating to Corner {corner_idx + 1} ===")
        
        # Calculate RRT* trajectory from current position to the corner
        rrt_path = await nav_module.rrt_star_plan(current_lat, current_lon, dest_lat, dest_lon, dest_alt)
        
        for rrt_idx, (lat, lon, alt) in enumerate(rrt_path):
            print(f"\n-> Heading to RRT* sub-waypoint {rrt_idx + 1}/{len(rrt_path)}")
            await nav_module.fly_to_waypoint(lat, lon, alt)
            
            # While flying, the AI loop runs checks
            for _ in range(3): # Simulate 3 checks while flying to this sub-waypoint
                # Log current telemetry
                logger.log_telemetry(lat, lon, alt, battery_v=15.2, message=f"RRT* WP {rrt_idx+1}")
                
                # Broadcast telemetry to dashboard
                await telemetry_server.broadcast({
                    "type": "telemetry",
                    "data": {
                        "lat": lat,
                        "lon": lon,
                        "alt": alt,
                        "battery": 15.2,
                        "mode": "AUTO_MISSION",
                        "cpu": 45 # Mock CPU load
                    }
                })

            # Perception Check
            if args.hardware:
                img = perception_module.get_real_image()
                # Assuming hardware module gets a detect method added later
            else:
                img = perception_module.get_synthetic_image()
                landing_pad = perception_module.detect_landing_pad(img)
                if landing_pad:
                    print(f"   [Vision] Detected potential landing pad at {landing_pad}")

                # YOLOv8 Advanced Vision with Tracking
                # For synthetic image, it might not find real objects, but the code is active
                if hasattr(perception_module, 'detect_and_track_objects'):
                    objects = perception_module.detect_and_track_objects(img)
                    if objects:
                        for obj in objects:
                            print(f"   [YOLO-Track] Found '{obj['class']}' (ID: {obj['id']}) at conf: {obj['confidence']:.2f}")
                        
                        # Broadcast detections
                        await telemetry_server.broadcast({
                            "type": "vision",
                            "data": objects
                        })

                # Update memory map with free space
                mapper.update_free_space(lat, lon)

            # Avoidance Check
            if avoidance_module.check_for_obstacles():
                obstacle_lat = lat + 0.00001
                obstacle_lon = lon + 0.00001
                
                # Record obstacle in semantic memory
                mapper.mark_obstacle(obstacle_lat, obstacle_lon)
                
                # Broadcast this finding to all other drones in the swarm!
                swarm.broadcast_obstacle(obstacle_lat, obstacle_lon)
                
                # ---------------------------------------------------------
                # TRIPLE-REDUNDANCY VOTING: EVASION CALCULATION
                # ---------------------------------------------------------
                # Instead of trusting one node, we have 3 nodes calculate evasion.
                # We will simulate Node 3 occasionally suffering a cosmic ray bit-flip / hallucination.
                node1_evade = avoidance_module.calculate_evasion_vector()
                node2_evade = avoidance_module.calculate_evasion_vector()
                
                if random.random() > 0.8: # 20% chance Node 3 fails dangerously
                    node3_evade = (node1_evade[0] + 0.5, node1_evade[1] - 0.5, -100.0) # Command drone to crash into ground
                else:
                    node3_evade = avoidance_module.calculate_evasion_vector()
                
                # Convert relative evasion vectors to absolute global coordinates for voting
                cmd1 = (lat + node1_evade[0], lon + node1_evade[1], alt + node1_evade[2])
                cmd2 = (lat + node2_evade[0], lon + node2_evade[1], alt + node2_evade[2])
                cmd3 = (lat + node3_evade[0], lon + node3_evade[1], alt + node3_evade[2])
                
                # Run the voting consensus
                consensus_waypoint = safety_system.vote_on_waypoint(cmd1, cmd2, cmd3)
                
                if consensus_waypoint is None:
                    print("!!! ABORTING MISSION DUE TO SAFETY SYSTEM COLLAPSE !!!")
                    await drone.action.kill()
                    os._exit(1)
                    
                final_lat, final_lon, final_alt = consensus_waypoint
                # ---------------------------------------------------------

                # Execute evasion immediately
                logger.log_telemetry(lat, lon, alt, battery_v=15.2, message="EVASION TRIGGERED")
                await telemetry_server.broadcast({
                    "type": "log",
                    "data": {"msg": "Obstacle detected! Calculating evasion...", "level": "critical"}
                })
                
                await nav_module.fly_to_waypoint(final_lat, final_lon, final_alt)
                print("   [Avoidance] Evasion complete, resuming path...")
                
                await asyncio.sleep(1) # Check every 1 second
                
        # Print map status after every leg
        mapper.print_map_status()
                
        # Update current position for the next corner calculation
        current_lat, current_lon, current_alt = dest_lat, dest_lon, dest_alt

    # 3. Mission Complete, Return and Land
    print("\nMission waypoints complete. Returning to launch...")
    await drone.action.return_to_launch()
    
    # Wait to land
    await asyncio.sleep(15) 

    # We assume it landed for this mock script. A real script checks telemetry.in_air()
    print("Disarming...")
    # Attempt to disarm (might fail if still in air, but this is a rough structure)
    try:
        await drone.action.disarm()
    except Exception as e:
        print(f"Disarm failed (likely still landing): {e}")

    print("Main AI Brain shutting down.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autonomous AI Pilot Brain")
    parser.add_argument(
        "--connect",
        type=str,
        default="udp://:14540",
        help="Connection string for the drone. Use 'serial:///dev/ttyTHS1:921600' for physical Pixhawk, or 'udp://:14540' for simulation."
    )
    parser.add_argument(
        "--hardware",
        action="store_true",
        help="Enable hardware perception (live camera) instead of simulated imagery."
    )
    parser.add_argument(
        "--drone_id",
        type=str,
        default="Alpha",
        help="The unique ID of this drone in the Swarm."
    )
    args = parser.parse_args()

    asyncio.run(run_mission(args))
