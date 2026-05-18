"""
Fix #21: Unit tests for EnvelopeProtectionSystem \u2014 the most safety-critical component.
Run with: pytest tests/test_envelope_protection.py -v
"""
import sys
import os
import asyncio
import pytest

# sys.path is managed by conftest.py
from envelope_protection import EnvelopeProtectionSystem


class MockLogger:
    """Minimal logger stub for testing."""
    def __init__(self):
        self.logs = []
    def log_telemetry(self, *args, **kwargs):
        self.logs.append(kwargs)

class MockDrone:
    """Stub that records commands instead of sending them to a real drone."""
    def __init__(self):
        self.last_goto = None
    class action:
        @staticmethod
        async def goto_location(lat, lon, alt, yaw):
            pass

class MockDroneRecorder:
    """Records the FINAL clamped command sent to the flight controller."""
    def __init__(self):
        self.last_lat = None
        self.last_lon = None
        self.last_alt = None
    class action:
        pass

# Home position for all tests
HOME_LAT = 40.7128  # New York City (not Zurich \u2014 proves dynamic geofence works)
HOME_LON = -74.0060


@pytest.fixture
def envelope():
    logger = MockLogger()
    return EnvelopeProtectionSystem(logger=logger, home_lat=HOME_LAT, home_lon=HOME_LON)


# ============================================================
# Test: Dynamic Geofence Calculation (Fix #10 validation)
# ============================================================

def test_geofence_is_relative_to_home(envelope):
    """Geofence must be calculated around the actual home position, not Zurich."""
    assert abs(envelope.GEOFENCE_MIN_LAT - (HOME_LAT - 0.02)) < 0.0001
    assert abs(envelope.GEOFENCE_MAX_LAT - (HOME_LAT + 0.02)) < 0.0001
    assert abs(envelope.GEOFENCE_MIN_LON - (HOME_LON - 0.02)) < 0.0001
    assert abs(envelope.GEOFENCE_MAX_LON - (HOME_LON + 0.02)) < 0.0001


# ============================================================
# Test: Altitude Clamping
# ============================================================

@pytest.mark.asyncio
async def test_altitude_above_ceiling_is_clamped(envelope):
    """AI cannot command altitude above MAX_ALTITUDE_M."""
    commands = []
    class RecordingDrone:
        class action:
            @staticmethod
            async def goto_location(lat, lon, alt, yaw):
                commands.append(alt)

    await envelope.safe_goto_location(RecordingDrone(), HOME_LAT, HOME_LON, 99999.0, 0.0)
    assert commands[0] == envelope.MAX_ALTITUDE_M, f"Expected alt clamped to {envelope.MAX_ALTITUDE_M}"


@pytest.mark.asyncio
async def test_altitude_below_floor_is_clamped(envelope):
    """AI cannot command altitude below MIN_ALTITUDE_M during cruise."""
    commands = []
    class RecordingDrone:
        class action:
            @staticmethod
            async def goto_location(lat, lon, alt, yaw):
                commands.append(alt)

    await envelope.safe_goto_location(RecordingDrone(), HOME_LAT, HOME_LON, 1.0, 0.0)
    assert commands[0] == envelope.MIN_ALTITUDE_M, f"Expected alt clamped to {envelope.MIN_ALTITUDE_M}"


@pytest.mark.asyncio
async def test_altitude_floor_disabled_in_landing_mode(envelope):
    """Altitude floor must be disabled when set_landing_mode(True) is called."""
    commands = []
    class RecordingDrone:
        class action:
            @staticmethod
            async def goto_location(lat, lon, alt, yaw):
                commands.append(alt)

    envelope.set_landing_mode(True)
    await envelope.safe_goto_location(RecordingDrone(), HOME_LAT, HOME_LON, 1.0, 0.0)
    # In landing mode, low altitude should NOT be clamped to MIN_ALTITUDE_M
    assert commands[0] == 1.0, "Landing mode should allow sub-MIN altitude commands"


# ============================================================
# Test: Geofence Clamping
# ============================================================

@pytest.mark.asyncio
async def test_lat_exceeding_geofence_is_clamped(envelope):
    """Latitude that breaches geofence must be clamped to the boundary."""
    commands = []
    class RecordingDrone:
        class action:
            @staticmethod
            async def goto_location(lat, lon, alt, yaw):
                commands.append((lat, lon))

    far_lat = HOME_LAT + 10.0  # Way outside geofence
    await envelope.safe_goto_location(RecordingDrone(), far_lat, HOME_LON, 100.0, 0.0)
    clamped_lat, _ = commands[0]
    assert clamped_lat <= envelope.GEOFENCE_MAX_LAT


@pytest.mark.asyncio
async def test_valid_command_passes_through_unchanged(envelope):
    """A safe command within all limits must pass through without modification."""
    commands = []
    class RecordingDrone:
        class action:
            @staticmethod
            async def goto_location(lat, lon, alt, yaw):
                commands.append((lat, lon, alt))

    safe_alt = 100.0
    await envelope.safe_goto_location(RecordingDrone(), HOME_LAT, HOME_LON, safe_alt, 0.0)
    lat_r, lon_r, alt_r = commands[0]
    assert lat_r == HOME_LAT
    assert lon_r == HOME_LON
    assert alt_r == safe_alt
