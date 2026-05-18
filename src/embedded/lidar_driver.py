try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False

class LidarDriver:
    """
    Embedded UART driver for TF-Mini / TF-Luna LiDAR modules.
    Provides precise distance to the ground for terrain clearance.
    """
    def __init__(self, port='/dev/ttyS0', baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.ser = None
        self.initialized = False
        
        if not SERIAL_AVAILABLE:
            raise ImportError("pyserial not available. Hardware LiDAR cannot be initialized.")
            
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=0.1)
            self.initialized = True
        except Exception as e:
            raise ConnectionError(f"Failed to connect to LiDAR at {port}: {e}")

    def read_distance(self):
        """
        Reads the distance protocol packet from TF-LiDAR.
        Returns distance in meters, or None.
        """
        if not self.initialized:
            return None
            
        try:
            # TF-Luna Frame Format: 0x59 0x59 Dist_L Dist_H Strength_L Strength_H Temp_L Temp_H Checksum
            while self.ser.in_waiting >= 9:
                if self.ser.read() == b'\x59':
                    if self.ser.read() == b'\x59':
                        dist_l = self.ser.read()[0]
                        dist_h = self.ser.read()[0]
                        # Discard strength and temp
                        self.ser.read(4) 
                        checksum = self.ser.read()[0] # We ignore checksum verification for simplicity here
                        
                        distance_cm = dist_l + (dist_h << 8)
                        return distance_cm / 100.0 # Convert to meters
            return None
        except Exception as e:
            return None

    def close(self):
        if self.ser:
            self.ser.close()
