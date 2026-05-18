try:
    import serial
    import pynmea2
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False

class GPSDriver:
    """
    Embedded UART driver for standard GNSS/GPS modules (e.g., u-blox NEO-M8N).
    Parses NMEA sentences for Lat, Lon, and Altitude.
    """
    def __init__(self, port='/dev/ttyTHS1', baudrate=9600):
        self.port = port
        self.baudrate = baudrate
        self.ser = None
        self.initialized = False
        
        if not SERIAL_AVAILABLE:
            raise ImportError("pyserial or pynmea2 not available. Hardware GPS cannot be initialized.")
            
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=1.0)
            self.initialized = True
        except Exception as e:
            raise ConnectionError(f"Failed to connect to GPS at {port}: {e}")

    def read_position(self):
        """
        Reads lines from the serial port and parses GPGGA for position.
        Returns (lat, lon, alt) or None if no fix/no data.
        """
        if not self.initialized:
            return None
            
        try:
            # Read a few lines to ensure we catch a GPGGA or GNRMC sentence
            for _ in range(5):
                line = self.ser.readline().decode('ascii', errors='replace').strip()
                if line.startswith('$GPGGA') or line.startswith('$GNGGA'):
                    msg = pynmea2.parse(line)
                    if msg.gps_qual > 0: # 1=Fix, 2=DGPS
                        lat = msg.latitude
                        lon = msg.longitude
                        alt = msg.altitude
                        return lat, lon, alt
            return None
        except Exception as e:
            return None

    def close(self):
        if self.ser:
            self.ser.close()
