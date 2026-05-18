import asyncio
from mavsdk import System
from mavsdk.offboard import (OffboardError, PositionNedYaw)

class AegisMAVSDKController:
    """
    AEGIS AUTONOMY: Genuine MAVSDK Flight Controller Integration
    Replaces mock terminal output with real MAVLink protocol communication.
    Connects directly to PX4 Hardware or Gazebo SITL.
    """
    def __init__(self):
        # The main interface to the drone
        self.drone = System()
        # SITL (Simulator) usually communicates over UDP port 14540
        # A physical radio telemetry link might use serial e.g., "serial:///dev/ttyUSB0:57600"
        self.connection_url = "udp://:14540"

    async def connect(self):
        print(f"--- AEGIS MAVSDK: Connecting to drone at {self.connection_url} ---")
        await self.drone.connect(system_address=self.connection_url)

        print("Waiting for drone to connect...")
        async for state in self.drone.core.connection_state():
            if state.is_connected:
                print(f"-- Drone discovered with UUID: {state.uuid}")
                break

    async def check_health(self):
        print("Waiting for drone to have a global position estimate (GPS lock)...")
        async for health in self.drone.telemetry.health():
            if health.is_global_position_ok and health.is_home_position_ok:
                print("-- Global position estimate OK")
                break

    async def arm_and_takeoff(self, altitude=10.0):
        print("--- AEGIS MAVSDK: Arming Motors ---")
        await self.drone.action.arm()
        
        print(f"--- AEGIS MAVSDK: Taking off to {altitude} meters ---")
        await self.drone.action.set_takeoff_altitude(altitude)
        await self.drone.action.takeoff()
        
        # Wait a few seconds for the drone to reach altitude
        await asyncio.sleep(8)

    async def navigate_to_waypoint(self, north_m, east_m, down_m, yaw_deg):
        """
        Uses Offboard control to fly to a specific X, Y, Z coordinate (NED frame).
        North (X), East (Y), Down (Z - note that altitude is negative!).
        """
        print(f"--- AEGIS MAVSDK: Navigating to N:{north_m} E:{east_m} D:{down_m} ---")
        
        # Set an initial setpoint before starting offboard mode (required by PX4)
        await self.drone.offboard.set_position_ned(
            PositionNedYaw(0.0, 0.0, 0.0, 0.0))

        # Start Offboard Mode
        try:
            await self.drone.offboard.start()
        except OffboardError as error:
            print(f"Starting offboard mode failed with error code: {error._result.result}")
            print("Disarming and aborting...")
            await self.drone.action.disarm()
            return

        # Send the actual waypoint command
        await self.drone.offboard.set_position_ned(
            PositionNedYaw(north_m, east_m, down_m, yaw_deg))
            
        # Wait for the drone to physically fly there
        await asyncio.sleep(10)

        # Stop offboard mode
        print("--- AEGIS MAVSDK: Waypoint reached. Stopping Offboard mode. ---")
        await self.drone.offboard.stop()

    async def return_and_land(self):
        print("--- AEGIS MAVSDK: Triggering Return to Launch (RTL) ---")
        await self.drone.action.return_to_launch()
        
        # In a real mission, we would monitor the telemetry altitude to know when it lands.
        print("Drone is returning home and will land automatically.")

    async def run_mission(self):
        await self.connect()
        await self.check_health()
        await self.arm_and_takeoff(altitude=10.0)
        
        # Fly 50 meters North, 20 meters East, stay at 10m altitude (Down is -10)
        await self.navigate_to_waypoint(north_m=50.0, east_m=20.0, down_m=-10.0, yaw_deg=90.0)
        
        await self.return_and_land()

if __name__ == "__main__":
    controller = AegisMAVSDKController()
    
    # We use asyncio to run the asynchronous flight plan
    loop = asyncio.get_event_loop()
    loop.run_until_complete(controller.run_mission())
