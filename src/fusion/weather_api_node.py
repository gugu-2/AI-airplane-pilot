import time
import json
import random
import urllib.request

class WeatherNode:
    """
    AEGIS AUTONOMY: Live Aviation Weather Integration
    Connects to live METAR/Weather APIs to prevent the drone from taking off 
    into severe thunderstorms, high winds, or zero-visibility fog.
    """
    def __init__(self):
        # Drone GPS Coordinates (San Francisco)
        self.lat = 37.7749
        self.lon = -122.4194
        
        # Flight Safety Constraints (DoD/Commercial Standards)
        self.max_wind_knots = 25.0
        self.min_visibility_km = 3.0
        self.max_precip_mm = 5.0

    def mock_fetch_weather_api(self):
        """
        Simulates an API call to AviationWeather.gov or OpenWeatherMap.
        In production, we would use: 
        req = urllib.request.urlopen(f"https://api.openweathermap.org/data/2.5/weather?lat={self.lat}&lon={self.lon}&appid=YOUR_API_KEY")
        """
        print(f"[WEATHER API] Fetching live METAR data for coordinates: {self.lat}, {self.lon}...")
        time.sleep(1) # Simulate network delay
        
        # Generate a random weather profile (Clear, Windy, or Stormy)
        profile = random.choice(['CLEAR', 'GALE', 'THUNDERSTORM'])
        
        if profile == 'CLEAR':
            return {"wind_speed": 10.5, "visibility": 10.0, "precipitation": 0.0, "desc": "Clear Skies"}
        elif profile == 'GALE':
            return {"wind_speed": 35.0, "visibility": 8.0, "precipitation": 0.0, "desc": "High Winds"}
        else:
            return {"wind_speed": 20.0, "visibility": 1.5, "precipitation": 12.0, "desc": "Heavy Rain/Fog"}

    def pre_flight_weather_check(self):
        """Evaluates live weather against the drone's physical aerodynamic limits."""
        weather = self.mock_fetch_weather_api()
        
        print("\n===========================================")
        print("    PRE-FLIGHT WEATHER METRICS")
        print("===========================================")
        print(f"Condition     : {weather['desc']}")
        print(f"Wind Speed    : {weather['wind_speed']} knots (Max: {self.max_wind_knots})")
        print(f"Visibility    : {weather['visibility']} km (Min: {self.min_visibility_km})")
        print(f"Precipitation : {weather['precipitation']} mm/hr (Max: {self.max_precip_mm})")
        print("===========================================\n")
        
        # Safety Logic
        if weather['wind_speed'] > self.max_wind_knots:
            print("[MISSION ABORT] [REJECTED] Wind shear exceeds aerodynamic limits. Drone will stall.")
            return False
            
        if weather['visibility'] < self.min_visibility_km:
            print("[MISSION ABORT] [REJECTED] Visibility too low for YOLOv8 visual servoing.")
            return False
            
        if weather['precipitation'] > self.max_precip_mm:
            print("[MISSION ABORT] [REJECTED] Heavy rain detected. Risk of ESC short-circuit.")
            return False

        print("[MISSION AUTHORIZED] [OK] Weather is GREEN. Sending launch clearance to MAVSDK.")
        return True

if __name__ == "__main__":
    node = WeatherNode()
    
    # Run the check 3 times to see different weather profiles
    for i in range(1, 4):
        print(f"\n>>> INITIATING PRE-FLIGHT CHECK {i}/3")
        is_safe = node.pre_flight_weather_check()
        time.sleep(2)
