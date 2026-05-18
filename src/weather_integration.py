import random
import time

class WeatherSensor:
    """
    Simulates a connection to a live Meteorological API (like METAR or OpenWeather).
    Provides wind speed, direction, and visibility data.
    """
    def __init__(self):
        # Baseline weather
        self.wind_speed_knots = 5.0
        self.wind_dir_deg = 270.0 # Wind coming from the West
        self.condition = "CLEAR"
        self.temperature_c = 22.0
        self.last_update = time.time()

    def get_live_weather(self):
        """
        Polls the 'API' and slightly shifts weather patterns over time.
        """
        current_time = time.time()
        if current_time - self.last_update > 2.0:
            # Shift weather slightly
            self.wind_speed_knots += random.uniform(-0.5, 0.5)
            # R16 FIX: Clamp here so check_weather_alarm also sees a valid value
            self.wind_speed_knots = max(0.0, self.wind_speed_knots)
            self.wind_dir_deg = (self.wind_dir_deg + random.uniform(-5.0, 5.0)) % 360
            
            # 5% chance of sudden storm microburst if not already storming
            if self.condition == "CLEAR" and random.random() < 0.05:
                self.condition = "STORM"
                self.wind_speed_knots = 35.0 # Dangerous winds
                
            # 5% chance storm clears
            elif self.condition == "STORM" and random.random() < 0.05:
                self.condition = "CLEAR"
                self.wind_speed_knots = 10.0
                
            self.last_update = current_time
            
        return {
            "wind_speed_knots": max(0.0, round(self.wind_speed_knots, 1)),
            "wind_dir_deg": int(self.wind_dir_deg),
            "condition": self.condition,
            "temperature_c": self.temperature_c
        }

    def check_weather_alarm(self):
        """
        Returns True if the weather is too dangerous for flight.
        """
        # FAA standard limit for many small drones is ~25 knots
        if self.wind_speed_knots > 25.0 or self.condition == "STORM":
            return True, f"SEVERE WEATHER: {self.condition}, {self.wind_speed_knots:.1f} KT WIND"
        return False, None
