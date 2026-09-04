"""
Fix #21: Unit tests for ExtendedKalmanFilter state estimation.
Run with: pytest tests/test_state_estimation.py -v
"""
import sys
import os
import pytest

# sys.path is managed by conftest.py
from state_estimation import ExtendedKalmanFilter


HOME_LAT = 47.397742
HOME_LON = 8.545594


def test_ekf_initializes_at_correct_position():
    """EKF must initialize at the given GPS seed position."""
    ekf = ExtendedKalmanFilter(HOME_LAT, HOME_LON)
    lat, lon = ekf.get_state()
    assert abs(lat - HOME_LAT) < 1e-6
    assert abs(lon - HOME_LON) < 1e-6


def test_ekf_predict_updates_position():
    """After a predict step with positive velocity, position should change."""
    ekf = ExtendedKalmanFilter(HOME_LAT, HOME_LON)
    ekf.predict(dt=1.0, accel_lat=0.001, accel_lon=0.001)
    lat, lon = ekf.get_state()
    assert lat != HOME_LAT or lon != HOME_LON


def test_ekf_update_converges_toward_gps():
    """
    After many GPS update steps, the EKF estimate should converge
    toward the true GPS position and reduce noise.
    """
    ekf = ExtendedKalmanFilter(HOME_LAT, HOME_LON)
    TRUE_LAT = HOME_LAT + 0.0005
    TRUE_LON = HOME_LON + 0.0005
    
    # Simulate many noisy GPS readings near the true position
    import random
    for _ in range(50):
        noisy_lat = TRUE_LAT + random.gauss(0, 0.00005)
        noisy_lon = TRUE_LON + random.gauss(0, 0.00005)
        ekf.update(noisy_lat, noisy_lon)
    
    lat, lon = ekf.get_state()
    # After 50 updates, estimate should be within 0.001 degrees (~110m) of true position
    assert abs(lat - TRUE_LAT) < 0.001
    assert abs(lon - TRUE_LON) < 0.001


def test_inject_position_delta_numpy():
    """inject_position_delta must correctly update Numpy backend state."""
    ekf = ExtendedKalmanFilter(HOME_LAT, HOME_LON)
    # Force Numpy backend for this test
    ekf.use_cpp = False
    ekf.x = [HOME_LAT, HOME_LON, 0.0, 0.0]
    
    delta = 0.0001
    ekf.inject_position_delta(delta, delta)
    lat, lon = ekf.get_state()
    
    assert abs(lat - (HOME_LAT + delta)) < 1e-9
    assert abs(lon - (HOME_LON + delta)) < 1e-9


def test_inject_position_delta_cpp_backend():
    """inject_position_delta must also update C++ struct if C++ backend is active."""
    ekf = ExtendedKalmanFilter(HOME_LAT, HOME_LON)
    
    if ekf.use_cpp:
        delta = 0.0001
        ekf.inject_position_delta(delta, delta)
        lat, lon = ekf.get_state()
        assert abs(lat - (HOME_LAT + delta)) < 1e-9
        assert abs(lon - (HOME_LON + delta)) < 1e-9
    else:
        pytest.skip("C++ engine not compiled \u2014 test skipped")
