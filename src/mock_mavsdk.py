import asyncio

class MockAction:
    def __init__(self, sys):
        self.sys = sys

    async def arm(self):
        print("[MOCK MAVSDK] Drone armed.")
        await asyncio.sleep(1)

    async def takeoff(self):
        print("[MOCK MAVSDK] Drone taking off.")
        self.sys.telemetry.current_alt = 488.0 + 10.0
        self.sys.telemetry.is_flying = True
        await asyncio.sleep(1)

    async def land(self):
        print("[MOCK MAVSDK] Drone landing.")
        self.sys.telemetry.current_alt = 488.0
        self.sys.telemetry.is_flying = False
        await asyncio.sleep(1)
        
    async def disarm(self):
        print("[MOCK MAVSDK] Drone disarmed.")
        await asyncio.sleep(1)

    async def return_to_launch(self):
        print("[MOCK MAVSDK] Returning to launch point.")
        # R11 FIX: Return to actual spawn position, not hardcoded Zurich
        self.sys.telemetry.current_lat = self.sys.home_lat
        self.sys.telemetry.current_lon = self.sys.home_lon
        await asyncio.sleep(1)
        await self.land()

    async def goto_location(self, lat, lon, alt, yaw):
        print(f"[MOCK MAVSDK] Flying to lat: {lat:.6f}, lon: {lon:.6f}, alt: {alt:.2f}")
        # Teleport to location after 2 seconds for mock purposes
        await asyncio.sleep(2)
        self.sys.telemetry.current_lat = lat
        self.sys.telemetry.current_lon = lon
        self.sys.telemetry.current_alt = alt

class MockCore:
    async def connection_state(self):
        class State:
            is_connected = True
        yield State()

class MockTelemetry:
    def __init__(self):
        import random
        self.current_lat = 47.397742 + random.uniform(-0.0005, 0.0005)
        self.current_lon = 8.545594 + random.uniform(-0.0005, 0.0005)
        self.current_alt = 488.0
        self.is_flying = False

    async def health(self):
        class Health:
            is_global_position_ok = True
            is_home_position_ok = True
        yield Health()
        
    async def in_air(self):
        while True:
            yield self.is_flying
            await asyncio.sleep(0.5)

    async def imu(self):
        class IMU:
            def __init__(self):
                class Accel:
                    def __init__(self):
                        import random
                        # Mock forward acceleration
                        self.x = random.uniform(-0.1, 0.1) 
                        self.y = random.uniform(-0.1, 0.1)
                        self.z = -9.81
                self.acceleration_frd = Accel()
                
        while True:
            yield IMU()
            await asyncio.sleep(0.1) # 10Hz IMU

    async def position(self):
        class Position:
            def __init__(self, lat, lon, alt):
                self.latitude_deg = lat
                self.longitude_deg = lon
                self.absolute_altitude_m = alt
                
        while True:
            import random
            # Intentionally inject violent Gaussian noise to simulate bad GPS bounce
            noisy_lat = self.current_lat + random.gauss(0, 0.0002) 
            noisy_lon = self.current_lon + random.gauss(0, 0.0002)
            yield Position(noisy_lat, noisy_lon, self.current_alt)
            await asyncio.sleep(0.5) # 2Hz GPS

class MockSystem:
    def __init__(self):
        self.telemetry = MockTelemetry()
        # R11 FIX: Store the actual spawn position so return_to_launch uses the real home
        self.home_lat = self.telemetry.current_lat
        self.home_lon = self.telemetry.current_lon
        self.core = MockCore()
        self.action = MockAction(self)

    async def connect(self, system_address="udp://:14540"):
        print(f"[MOCK MAVSDK] Connected to virtual simulation at {system_address}")
        await asyncio.sleep(1)
