import asyncio
import argparse
import os
import time
import math
import random

# MAVSDK import is deferred to runtime based on --hardware flag.

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
from envelope_protection import EnvelopeProtectionSystem
from state_estimation import ExtendedKalmanFilter
from adsb_awareness import ADSBSensor
from weather_integration import WeatherSensor

# Global state for ATC overrides
atc_override_intent = None

# A2 FIX: Stores the drone reference so the ATC callback can actually command it.
# This is set up inside run_mission() once the drone object is ready.
_drone_ref = None
_current_pos_ref = {'lat': 0.0, 'lon': 0.0, 'alt': 0.0}

async def handle_atc_override(parsed_intent):
    """A2 FIX: Actually executes ATC commands instead of just printing the intent."""
    intent = parsed_intent.get("intent")
    print(f"\n[AI Brain] CRITICAL: Executing ATC Override -> {intent}")

    if _drone_ref is None:
        print("[ATC] Drone not ready, command ignored.")
        return

    try:
        if intent == "HOLD_POSITION":
            # Hold: fly to current GPS position (hover in place)
            lat = _current_pos_ref['lat']
            lon = _current_pos_ref['lon']
            alt = _current_pos_ref['alt']
            await _drone_ref.action.goto_location(lat, lon, alt, 0.0)
            print(f"[ATC] Holding position at ({lat:.6f}, {lon:.6f})")

        elif intent == "ALTITUDE_CHANGE":
            # A2 FIX: Convert ATC feet to meters and fly there
            target_ft = parsed_intent.get("target_altitude", 500)
            target_m = target_ft * 0.3048
            lat = _current_pos_ref['lat']
            lon = _current_pos_ref['lon']
            print(f"[ATC] Altitude change: {target_ft} ft = {target_m:.1f} m")
            await _drone_ref.action.goto_location(lat, lon, target_m, 0.0)

        elif intent == "ABORT_LANDING":
            print("[ATC] Go-around commanded. Climbing to 100m.")
            lat = _current_pos_ref['lat']
            lon = _current_pos_ref['lon']
            await _drone_ref.action.goto_location(lat, lon, 100.0, 0.0)

        elif intent == "LANDING_CLEARANCE":
            print("[ATC] Landing clearance received. Initiating landing sequence.")
            await _drone_ref.action.land()

        elif intent == "TAKEOFF_CLEARANCE":
            print("[ATC] Takeoff clearance acknowledged.")
            # Takeoff is already in progress in the mission loop, just acknowledge.

    except Exception as e:
        print(f"[ATC] Failed to execute {intent}: {e}")

async def monitor_kill_switch(drone: System, logger):
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
                    logger.finalize_flight(status="PILOT_KILL_SWITCH_TRIGGERED")
                    logger.close()
                    # Force exit
                    os._exit(1)
    except Exception as e:
        # Mock system might not have RC status implemented
        pass

class MissionAbortException(Exception):
    """Raised to safely abort the mission and trigger the cleanup finally block."""
    def __init__(self, status="ABORTED", message=""):
        self.status = status
        self.message = message
        super().__init__(message)

async def emergency_shutdown(drone, logger, status, message):
    """Graceful emergency shutdown — never uses os._exit(). Always lets finally run."""
    print(f"\n!!! EMERGENCY SHUTDOWN: {message} !!!")
    try:
        await drone.action.kill()
    except Exception:
        pass
    raise MissionAbortException(status=status, message=message)

    connection_address = args.connect
    print("Initializing Main AI Brain...")
    
    # Conditionally import the real MAVSDK or the mock depending on the --hardware flag.
    # This was previously hardcoded to USE_MOCK = True which prevented real hardware from ever connecting.
    if args.hardware:
        print("--- Running in REAL HARDWARE Mode (MAVSDK) ---")
        from mavsdk import System
    else:
        print("--- Running in MOCK Simulation Mode ---")
        from mock_mavsdk import MockSystem as System
    
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
        mapper = SemanticMap(grid_size=200, resolution=0.5)
    
        if args.disable_ws:
            telemetry_server = None
            telemetry_task = None
        else:
            telemetry_server = TelemetryServer(port=args.ws_port)
            telemetry_server.on_atc_intent = handle_atc_override
            telemetry_task = asyncio.create_task(telemetry_server.start_server())
            background_tasks.append(telemetry_task)
            
        safety_system = TripleRedundancySystem(tolerance_meters=2.0)
    
        # Initialize Swarm Intelligence
        swarm = SwarmNetwork(drone_id=args.drone_id)
        swarm.attach_mapper(mapper)
        
        if telemetry_server:
            async def handle_swarm_telemetry(msg):
                await telemetry_server.broadcast({"type": "swarm_telemetry", "data": msg})
            swarm.on_swarm_telemetry = handle_swarm_telemetry

        kill_switch_task = asyncio.create_task(monitor_kill_switch(drone, logger))
        swarm_task = asyncio.create_task(swarm.listen_for_swarm())
        background_tasks += [kill_switch_task, swarm_task]

    # R1 FIX: Fetch real GPS position BEFORE initializing geofence and EKF.
    # Previously, home_lat/lon were 0 at envelope init time.
    print("Fetching GPS position before module initialization...")
    current_lat, current_lon, current_alt = 0.0, 0.0, 0.0
    async for position in drone.telemetry.position():
        current_lat = position.latitude_deg
        current_lon = position.longitude_deg
        current_alt = position.absolute_altitude_m
        print(f"[GPS] Home position locked: ({current_lat:.6f}, {current_lon:.6f})")
        break

    # Now safe to initialize envelope with real home coordinates
    envelope = EnvelopeProtectionSystem(
        logger=logger,
        telemetry_server=telemetry_server,
        home_lat=current_lat,
        home_lon=current_lon
    )
    nav_module = NavigationModule(drone, envelope=envelope)

    # A2 FIX: Expose drone reference to ATC callback so it can actually command the aircraft
    global _drone_ref, _current_pos_ref
    _drone_ref = drone
    _current_pos_ref = {'lat': current_lat, 'lon': current_lon, 'alt': current_alt}

    print("\n--- Starting Mission ---")
    print("Arming...")
    await drone.action.arm()
    
    print("Taking off...")
    await drone.action.takeoff()
    await asyncio.sleep(8)

    # Generate mission path
    square_corners = await nav_module.generate_square_waypoints(side_length=15.0)
    print(f"Generated {len(square_corners)} high-level corners for the mission.")

    print("[EKF] Initializing Extended Kalman Filter...")
    ekf = ExtendedKalmanFilter(initial_lat=current_lat, initial_lon=current_lon)
    
    adsb = ADSBSensor(anchor_lat=current_lat, anchor_lon=current_lon)
    weather = WeatherSensor()
    
    gps_lost = False
    mission_start_time = time.time()
    
    async def feed_ekf_gps():
        nonlocal gps_lost
        async for pos in drone.telemetry.position():
            if args.jam_gps and (time.time() - mission_start_time) > 15:
                if not gps_lost:
                    print("\n[CRITICAL ALARM] GPS SIGNAL LOST (MILITARY JAMMING DETECTED)!")
                    print("[SYSTEM] Switching to Visual Odometry (Dead Reckoning SLAM).")
                    gps_lost = True
                continue
            if not gps_lost:
                ekf.update(pos.latitude_deg, pos.longitude_deg)
            
    async def feed_ekf_imu():
        try:
            async for imu in drone.telemetry.imu():
                ekf.predict(0.1, imu.acceleration_frd.x, imu.acceleration_frd.y)
        except Exception:
            pass
            
    ekf_gps_task = asyncio.create_task(feed_ekf_gps())
    ekf_imu_task = asyncio.create_task(feed_ekf_imu())
    # R3 FIX: Add EKF tasks to the cleanup list so they are cancelled on shutdown
    background_tasks += [ekf_gps_task, ekf_imu_task]

    mapper.set_anchor(current_lat, current_lon)

    # Execute mission path
    for corner_idx, (dest_lat, dest_lon, dest_alt) in enumerate(square_corners):
        print(f"\n=== Navigating to Corner {corner_idx + 1} ===")
        
        if args.use_rl:
            path = await nav_module.cognitive_rl_plan(current_lat, current_lon, dest_lat, dest_lon, dest_alt, mapper)
        else:
            # Fix #6: Renamed from rrt_star_plan to honest linear_waypoint_plan
            path = await nav_module.linear_waypoint_plan(current_lat, current_lon, dest_lat, dest_lon, dest_alt, mapper)
        
        for idx, (lat, lon, alt) in enumerate(path):
            if args.use_rl:
                print(f"\n-> Heading to DQN sub-waypoint {idx + 1}/{len(path)}")
            else:
                print(f"\n-> Heading to linear sub-waypoint {idx + 1}/{len(path)}")
                
            await nav_module.fly_to_waypoint(lat, lon, alt)
            
            # While flying, the AI loop runs checks
            for _ in range(3): # Simulate 3 checks while flying to this sub-waypoint
                wp_label = "DQN WP" if args.use_rl else "WP"
                logger.log_telemetry(lat, lon, alt, battery_v=15.2, message=f"{wp_label} {idx+1}")
                
                # Broadcast our own telemetry to the Swarm!
                swarm.broadcast_telemetry(lat, lon, alt)
                
                # Broadcast telemetry to our local dashboard
                if telemetry_server:
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
            else:
                img = perception_module.get_synthetic_image()
                
            if img is not None:
                landing_pad = perception_module.detect_landing_pad(img)
                if landing_pad:
                    print(f"   [Vision] Detected potential landing pad at {landing_pad}")

                # YOLOv8 Advanced Vision with Tracking
                objects = perception_module.detect_and_track_objects(img)
                if objects:
                    for obj in objects:
                        print(f"   [YOLO-Track] Found '{obj['class']}' (ID: {obj['id']}) at conf: {obj['confidence']:.2f}")
                    if telemetry_server:
                        await telemetry_server.broadcast({
                            "type": "vision",
                            "data": objects
                        })
            else:
                objects = []  # No image = no detections

            # Fetch Embedded Hardware Sensor Data (including Visual Odometry)
            # Try to grab a real frame if we have the hardware camera
            frame = None
            if hasattr(perception_module, 'get_real_image'):
                frame = perception_module.get_real_image()
                
            if hasattr(perception_module, 'read_embedded_sensors'):
                hardware_data = perception_module.read_embedded_sensors(
                    current_frame=frame,
                    altitude_m=current_alt,
                    vision_objects_count=len(objects) if objects else 0
                )
                logger.log_hardware_telemetry(
                    pitot_airspeed=hardware_data["pitot_airspeed"],
                    lidar_distance=hardware_data["lidar_distance"],
                    vision_targets_count=hardware_data["vision_targets_count"]
                )
                
                # GPS-Denied Dead Reckoning Integration (Fix #3: uses inject_position_delta)
                if gps_lost:
                    vo_vx = hardware_data.get("vo_velocity_x", 0.0)
                    vo_vy = hardware_data.get("vo_velocity_y", 0.0)
                    # Convert m/s to degrees (Approx 111,320 meters per degree lat)
                    lat_shift = (vo_vy * 0.1) / 111320.0
                    lon_shift = (vo_vx * 0.1) / (111320.0 * math.cos(math.radians(current_lat)))
                    # Use the safe inject method that works with both C++ and Numpy backends
                    ekf.inject_position_delta(lat_shift, lon_shift)

            # --- 2. AVIATION AWARENESS (ADS-B & WEATHER) ---
            live_traffic = adsb.get_live_traffic()
            tcas_alert, threat_callsign = adsb.check_tcas_alarm(current_lat, current_lon, current_alt)
            if tcas_alert:
                print(f"\n[TCAS ALARM] TRAFFIC ALERT: {threat_callsign} IS DANGEROUSLY CLOSE. DESCENDING!")
                # Fix #19: Immediately command a descent — don't wait for next waypoint
                safe_descent_alt = max(current_alt - 5.0, envelope.MIN_ALTITUDE_M + 5)
                await nav_module.fly_to_waypoint(current_lat, current_lon, safe_descent_alt)
                
            live_weather = weather.get_live_weather()
            weather_alert, wx_msg = weather.check_weather_alarm()
            if weather_alert:
                print(f"\n[WEATHER ALARM] {wx_msg}. INITIATING RETURN TO LAUNCH!")
                # Fix #18: Actually trigger RTL instead of just printing a warning
                logger.log_telemetry(current_lat, current_lon, current_alt, battery_v=15.2, message=f"WEATHER_RTL: {wx_msg}", level="critical")
                envelope.set_landing_mode(True)
                await drone.action.return_to_launch()
                raise MissionAbortException(status="WEATHER_RTL", message=wx_msg)
                
            if telemetry_server:
                await telemetry_server.broadcast({
                    "type": "aviation_awareness",
                    "data": {
                        "traffic": live_traffic,
                        "weather": live_weather,
                        "tcas_alert": tcas_alert,
                        "weather_alert": weather_alert
                    }
                })

            # Update memory map with free space
            mapper.update_free_space(lat, lon)

            # A2 FIX: Keep ATC callback position in sync
            _current_pos_ref['lat'] = lat
            _current_pos_ref['lon'] = lon
            _current_pos_ref['alt'] = alt

            # --- OBSTACLE AVOIDANCE (A3 + A4 FIX) ---
            # A4: Pass real YOLO detections and LiDAR distance instead of coin flip
            lidar_m = hardware_data.get('lidar_distance') if 'hardware_data' in dir() else None
            yolo_objs = objects if 'objects' in dir() and objects else []
            if avoidance_module.check_for_obstacles(detected_objects=yolo_objs, lidar_distance_m=lidar_m):
                logger.record_obstacle()
                # A3 FIX: Calculate real obstacle GPS using bearing + LiDAR distance
                # Previously: obstacle_lat = lat + 0.00001 (always 1.1m north — wrong)
                drone_heading = getattr(live_traffic[0], 'heading', 0) if live_traffic else 0
                obstacle_lat, obstacle_lon = avoidance_module.project_obstacle_gps(
                    lat, lon, drone_heading,
                    lidar_distance_m=lidar_m,
                    detected_objects=yolo_objs
                )
                print(f"   [Avoidance] Obstacle projected at ({obstacle_lat:.6f}, {obstacle_lon:.6f})")
                
                # Record obstacle in semantic memory
                mapper.mark_obstacle(obstacle_lat, obstacle_lon)
                
                # Broadcast this finding to all other drones in the swarm!
                swarm.broadcast_obstacle(obstacle_lat, obstacle_lon)
                
                # Fix #4: Added None guard — telemetry_server can be None with --disable_ws
                if telemetry_server:
                    await telemetry_server.broadcast({
                        "type": "map_state",
                        "data": mapper.get_obstacles()
                    })
                
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
                    logger.finalize_flight(status="CRITICAL_SAFETY_COLLAPSE")
                    logger.close()
                    os._exit(1)
                    
                final_lat, final_lon, final_alt = consensus_waypoint
                # ---------------------------------------------------------

                # Execute evasion immediately
                logger.log_telemetry(lat, lon, alt, battery_v=15.2, message="EVASION TRIGGERED", level="warning")
                logger.record_evasion()
                if telemetry_server:
                    await telemetry_server.broadcast({
                        "type": "log",
                        "data": {"msg": "Obstacle detected! Calculating evasion...", "level": "critical"}
                    })
                
                await nav_module.fly_to_waypoint(final_lat, final_lon, final_alt)
                print("   [Avoidance] Evasion complete, resuming path...")
                
                await asyncio.sleep(1) # Check every 1 second
                
        # Print map status after every leg
        mapper.print_map_status()
                
        # Update current position for the next corner calculation using the CLEAN EKF output
        current_lat, current_lon = ekf.get_state()
        current_alt = dest_alt # altitude isn't filtered in this MVP

    # The actual mission logic is complete. We are returning to launch.
    print("Mission waypoints complete. Returning to launch...")
    envelope.set_landing_mode(True)
    await drone.action.return_to_launch()
    
    # Wait to land safely using telemetry
    print("Waiting for drone to land...")
    # Replace sleep with an active polling loop checking if it has landed
    # Also listen for ATC go-around commands!
    global atc_override_intent
    async for in_air in drone.telemetry.in_air():
        if atc_override_intent == "ABORT_LANDING":
            print("!!! ATC COMMANDED ABORT LANDING. EXECUTING GO-AROUND !!!")
            logger.log_telemetry(0, 0, 0, battery_v=15.2, message="ATC ABORT LANDING", level="critical")
            await drone.action.takeoff() # Use takeoff or climb
            atc_override_intent = None
            await asyncio.sleep(5)
            print("Go-around complete. Attempting to land again...")
            await drone.action.return_to_launch()
            
        if not in_air:
            print("-- Landed safely.")
            break
        await asyncio.sleep(1)

    print("Disarming...")
    try:
        await drone.action.disarm()
    except Exception as e:
        print(f"Disarm failed: {e}")

    logger.finalize_flight(status="SUCCESS")
    logger.close()
    
    except MissionAbortException as e:
        print(f"[MISSION ABORTED] Status: {e.status}")
        if logger:
            logger.finalize_flight(status=e.status)
    except Exception as e:
        print(f"[FATAL ERROR] Unexpected exception: {e}")
        import traceback
        traceback.print_exc()
        if logger:
            logger.finalize_flight(status="FATAL_ERROR")
    finally:
        print("\n[Cleanup] Running graceful shutdown sequence...")
        # Cancel all background tasks
        for task in background_tasks:
            task.cancel()
        # Release hardware resources (Fix #23)
        if perception_module and hasattr(perception_module, 'release'):
            perception_module.release()
        # Always close the black box DB (Fix #2)
        if logger:
            logger.close()
        print("[Cleanup] All resources released. Goodbye.")

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
    parser.add_argument(
        "--ws_port",
        type=int,
        default=8765,
        help="WebSocket port for the local React dashboard."
    )
    parser.add_argument(
        "--disable_ws",
        action="store_true",
        help="Disable the WebSocket server (useful for secondary drones)."
    )
    parser.add_argument(
        "--use_rl",
        action="store_true",
        help="Enable the PyTorch Deep Q-Network for cognitive path planning instead of mathematical RRT*."
    )
    parser.add_argument(
        "--jam_gps",
        action="store_true",
        help="Simulate a GPS denial attack after 15 seconds to test Visual Odometry SLAM."
    )
    args = parser.parse_args()

    asyncio.run(run_mission(args))
