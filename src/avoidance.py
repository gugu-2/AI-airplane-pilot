import random

class ObstacleAvoidanceModule:
    def __init__(self):
        # In a real environment, this would process point clouds or depth maps.
        pass

    def check_for_obstacles(self) -> bool:
        """
        Simulates checking for an obstacle directly in the flight path.
        Returns True if an obstacle is detected, False otherwise.
        """
        # For simulation purposes, let's randomly detect an obstacle 5% of the time.
        # This allows us to test the evasion logic in the main loop.
        probability_of_obstacle = 0.05
        obstacle_detected = random.random() < probability_of_obstacle
        
        if obstacle_detected:
            print(">>> ALERT: Obstacle detected in flight path! <<<")
            
        return obstacle_detected

    def calculate_evasion_vector(self):
        """
        Calculates a simple vector (offset) to evade the obstacle.
        In reality, this involves complex trajectory planning.
        """
        print("Calculating evasion trajectory...")
        # For simplicity, we just decide to fly slightly higher and to the right
        # Added tiny random jitter so redundancy nodes don't vote on identical values
        alt_offset = 2.0 + random.uniform(-0.1, 0.1)
        lat_offset = 0.00002 + random.uniform(-0.000001, 0.000001)
        lon_offset = 0.00002 + random.uniform(-0.000001, 0.000001)
        
        return (lat_offset, lon_offset, alt_offset)
