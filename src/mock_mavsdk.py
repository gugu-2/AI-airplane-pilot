import asyncio

class MockAction:
    async def arm(self):
        print("[MOCK MAVSDK] Drone armed.")
        await asyncio.sleep(1)

    async def takeoff(self):
        print("[MOCK MAVSDK] Drone taking off.")
        await asyncio.sleep(1)

    async def land(self):
        print("[MOCK MAVSDK] Drone landing.")
        await asyncio.sleep(1)
        
    async def disarm(self):
        print("[MOCK MAVSDK] Drone disarmed.")
        await asyncio.sleep(1)

    async def return_to_launch(self):
        print("[MOCK MAVSDK] Returning to launch point.")
        await asyncio.sleep(1)

    async def goto_location(self, lat, lon, alt, yaw):
        print(f"[MOCK MAVSDK] Flying to lat: {lat:.6f}, lon: {lon:.6f}, alt: {alt:.2f}")
        await asyncio.sleep(2)

class MockCore:
    async def connection_state(self):
        class State:
            is_connected = True
        yield State()

class MockTelemetry:
    async def health(self):
        class Health:
            is_global_position_ok = True
            is_home_position_ok = True
        yield Health()
        
    async def in_air(self):
        yield False

    async def position(self):
        class Position:
            latitude_deg = 47.397742
            longitude_deg = 8.545594
            absolute_altitude_m = 488.0
        yield Position()

class MockSystem:
    def __init__(self):
        self.action = MockAction()
        self.core = MockCore()
        self.telemetry = MockTelemetry()

    async def connect(self, system_address="udp://:14540"):
        print(f"[MOCK MAVSDK] Connected to virtual simulation at {system_address}")
        await asyncio.sleep(1)
