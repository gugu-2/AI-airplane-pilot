import numpy as np
import matplotlib.pyplot as plt

class SemanticMap:
    def __init__(self, grid_size=100, resolution=1.0):
        """
        Initializes a 2D Occupancy Grid Map.
        grid_size: Number of cells in one dimension (e.g., 100x100 grid)
        resolution: Physical size of each cell in meters (e.g., 1.0 meters)
        """
        self.grid_size = grid_size
        self.resolution = resolution
        
        # Grid values: 0 (unknown), 1 (free space), -1 (obstacle)
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        
        # GPS anchor (takeoff point = center of grid)
        self.anchor_lat = None
        self.anchor_lon = None
        
        # R4 FIX: Keep a simple list of obstacle GPS coords for fast O(1) lookup
        # Previously mapper.obstacles did not exist, crashing the RL planner.
        self.obstacles = []  # list of {'lat': float, 'lon': float}
        
        print(f"[Mapper] Initialized {grid_size}x{grid_size} semantic occupancy grid.")

    def set_anchor(self, lat: float, lon: float):
        """Sets the center (0,0) of the grid to the drone's takeoff GPS location."""
        self.anchor_lat = lat
        self.anchor_lon = lon
        print(f"[Mapper] Grid anchor set to Lat: {lat}, Lon: {lon}")

    def _gps_to_grid(self, lat: float, lon: float):
        """Converts GPS coordinates to grid cell indices."""
        if self.anchor_lat is None or self.anchor_lon is None:
            return None, None
            
        # Rough approximation for simulation: 1 deg lat = 111,111 meters
        y_meters = (lat - self.anchor_lat) * 111111.0
        x_meters = (lon - self.anchor_lon) * (111111.0 * np.cos(np.radians(self.anchor_lat)))
        
        # Center the anchor in the middle of the grid
        center_idx = self.grid_size // 2
        
        x_idx = int(center_idx + (x_meters / self.resolution))
        y_idx = int(center_idx + (y_meters / self.resolution))
        
        return x_idx, y_idx

    def update_free_space(self, lat: float, lon: float):
        """Marks the drone's current location as free space."""
        x, y = self._gps_to_grid(lat, lon)
        if x is not None and y is not None and 0 <= x < self.grid_size and 0 <= y < self.grid_size:
            self.grid[y, x] = 1 # Mark as free space

    def mark_obstacle(self, lat: float, lon: float):
        """Marks a specific location as an obstacle in memory."""
        x, y = self._gps_to_grid(lat, lon)
        if x is not None and y is not None and 0 <= x < self.grid_size and 0 <= y < self.grid_size:
            self.grid[y, x] = -1
            # R4 FIX: Also append to the fast-access obstacles list
            self.obstacles.append({'lat': lat, 'lon': lon})
            print(f"   [Mapper] Obstacle logged at grid [{x}, {y}] | GPS ({lat:.6f}, {lon:.6f})")

    def print_map_status(self):
        """Prints basic statistics about the memory map."""
        obstacles = np.sum(self.grid == -1)
        free_space = np.sum(self.grid == 1)
        print(f"   [Mapper] Map Status: {obstacles} obstacles, {free_space} explored cells.")

    def get_obstacles(self):
        """
        Returns a list of {'lat', 'lon'} for all known obstacles.
        R12 FIX: Previously iterated the entire 40,000-cell grid on every call (O(n²)).
        Now returns the pre-built list directly in O(1).
        """
        return self.obstacles
