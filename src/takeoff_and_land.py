import asyncio
import os
# We force MOCK mode on Windows by default because MAVSDK server binary is missing.
USE_MOCK = True

if USE_MOCK or os.name == 'nt':
    print("\n--- Running in MOCK Simulation Mode ---")
    from mock_mavsdk import MockSystem as System
else:
    from mavsdk import System


async def run():
    print("Initializing Autonomous Brain...")
    
    # Initialize the MAVSDK System
    drone = System()
    
    # Connect to the local SITL instance. 
    # Usually, PX4 SITL listens on udp://:14540
    print("Connecting to drone...")
    await drone.connect(system_address="udp://:14540")

    # Wait for connection
    print("Waiting for drone to connect...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print(f"-- Connected to drone!")
            break

    # Check if the drone has a global position estimate
    print("Waiting for drone to have a global position estimate...")
    async for health in drone.telemetry.health():
        if health.is_global_position_ok and health.is_home_position_ok:
            print("-- Global position estimate OK")
            break

    print("Arming...")
    await drone.action.arm()

    print("Taking off...")
    await drone.action.takeoff()

    # Wait a few seconds for the drone to reach takeoff altitude
    await asyncio.sleep(10)

    print("Landing...")
    await drone.action.land()

    # Wait until the drone is on the ground
    async for in_air in drone.telemetry.in_air():
        if not in_air:
            print("-- Landed safely.")
            break
            
    # Disarm after landing
    print("Disarming...")
    await drone.action.disarm()
    print("Mission complete. Brain shutting down.")

if __name__ == "__main__":
    # Run the asyncio loop
    asyncio.run(run())
