import csv
import os
from datetime import datetime

class FlightLogger:
    def __init__(self, log_dir="logs"):
        """
        Initializes a CSV-based "Black Box" flight logger.
        """
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.filepath = os.path.join(log_dir, f"flight_log_{timestamp}.csv")
        
        print(f"[Logger] Initializing flight blackbox at {self.filepath}")
        
        # Write headers
        with open(self.filepath, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Timestamp", "Latitude", "Longitude", "Altitude_m", "Battery_V", "State_Message"])

    def log_telemetry(self, lat: float, lon: float, alt: float, battery_v: float, message: str = ""):
        """
        Logs a single row of telemetry.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        
        try:
            with open(self.filepath, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([timestamp, lat, lon, alt, battery_v, message])
        except Exception as e:
            print(f"[Logger] Failed to write log: {e}")
