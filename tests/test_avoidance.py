import pytest
import math
import numpy as np
from src.embedded.realsense_d435_driver import RealSenseD435Driver
from src.avoidance import ObstacleAvoidanceModule


def test_realsense_driver_simulation_mode():
    """Verifies that the RealSense driver initializes and generates simulated points when hardware is absent."""
    driver = RealSenseD435Driver()
    
    # Generate points (should return synthetic wall points)
    points = driver.get_pointcloud()
    assert len(points) > 0, "Point cloud should contain simulated points."
    
    # Verify shape
    if isinstance(points, np.ndarray):
        assert points.shape[1] == 3, "Points must be 3D coordinates (x, y, z)."
    else:
        assert len(points[0]) == 3, "Points must be 3D coordinates (x, y, z)."


def test_realsense_collision_detection_positive():
    """Verifies that the 3D bounding box triggers collision correctly when points are in range."""
    driver = RealSenseD435Driver()
    
    # Create a test point directly in front of the drone (Z=5.0m, X=0.0m, Y=0.0m)
    test_points = np.array([[0.0, 0.0, 5.0]])
    
    collision, dist = driver.check_collision(
        test_points,
        max_distance_m=10.0,
        box_width_m=2.0,
        box_height_m=2.0
    )
    assert collision is True, "Obstacle inside BBox should trigger collision."
    assert dist == 5.0, "Closest distance should be exactly 5.0m."


def test_realsense_collision_detection_negative_lateral():
    """Verifies that obstacles outside the lateral width of the box do not trigger collisions."""
    driver = RealSenseD435Driver()
    
    # Obstacle is placed far to the right (X=3.0m, Z=5.0m, Y=0.0m) - box width is 2.0m (half width is 1.0m)
    test_points = np.array([[3.0, 0.0, 5.0]])
    
    collision, dist = driver.check_collision(
        test_points,
        max_distance_m=10.0,
        box_width_m=2.0,
        box_height_m=2.0
    )
    assert collision is False, "Obstacle outside box width should be ignored."
    assert dist is None


def test_realsense_collision_detection_negative_depth():
    """Verifies that obstacles further than the max depth of the box do not trigger collisions."""
    driver = RealSenseD435Driver()
    
    # Obstacle is placed too far ahead (Z=15.0m, X=0.0m, Y=0.0m) - max range is 10.0m
    test_points = np.array([[0.0, 0.0, 15.0]])
    
    collision, dist = driver.check_collision(
        test_points,
        max_distance_m=10.0,
        box_width_m=2.0,
        box_height_m=2.0
    )
    assert collision is False, "Obstacle further than max range should be ignored."
    assert dist is None


def test_obstacle_avoidance_module_integration():
    """Verifies that the main avoidance module processes PointCloud data correctly."""
    module = ObstacleAvoidanceModule()
    
    # If the simulated RealSense points trigger a collision, check_for_obstacles should return True
    # (Simulated cloud has points at Z=6.0m or Z=12.0m. We mock the check to guarantee triggers)
    
    # Mock driver get_pointcloud to return a point directly in collision zone
    module.realsense.get_pointcloud = lambda: np.array([[0.1, -0.1, 4.0]])
    
    obstacle_detected = module.check_for_obstacles()
    assert obstacle_detected is True, "Avoidance module should detect collision from PointCloud."
    assert module.last_collision_distance == 4.0
    
    # Verify GPS projection of this obstacle
    drone_lat = 47.3977
    drone_lon = 8.5455
    drone_heading = 90.0 # Facing East
    
    obs_lat, obs_lon = module.project_obstacle_gps(drone_lat, drone_lon, drone_heading)
    
    # Since heading is 90 (East), longitude should increase while latitude remains very similar
    assert obs_lon > drone_lon, "Obstacle to the East should have a larger longitude."
    assert math.isclose(obs_lat, drone_lat, abs_tol=1e-4)
