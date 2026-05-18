try:
    from smbus2 import SMBus
    SMBUS_AVAILABLE = True
except ImportError:
    SMBUS_AVAILABLE = False
import time
import math

class PitotTubeDriver:
    """
    Embedded I2C driver for MS4525DO Airspeed sensor.
    Calculates Indicated Airspeed (IAS) from differential pressure.
    """
    def __init__(self, bus_num=1, address=0x28):
        self.bus_num = bus_num
        self.address = address
        self.bus = None
        self.initialized = False
        
        if not SMBUS_AVAILABLE:
            raise ImportError("smbus2 library not available. Hardware Pitot Tube cannot be initialized.")
            
        try:
            self.bus = SMBus(self.bus_num)
            # Perform a test read to ensure device is online
            self.bus.read_byte(self.address)
            self.initialized = True
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Pitot Tube at I2C bus {bus_num}, address {hex(address)}: {e}")

    def read_airspeed(self):
        """
        Reads differential pressure and converts to Indicated Airspeed (m/s).
        Returns None if read fails.
        """
        if not self.initialized:
            return None
            
        try:
            # Send a read command (usually just reading 4 bytes)
            data = self.bus.read_i2c_block_data(self.address, 0, 4)
            
            # Status is the top 2 bits of the first byte
            status = (data[0] & 0xC0) >> 6
            if status != 0:
                # 0 = Normal Operation, others indicate fault/stale data
                return None
                
            # 14-bit pressure data
            dp_raw = ((data[0] & 0x3F) << 8) | data[1]
            
            # Convert to actual pressure (psi)
            # Formula varies by exact MS4525DO model (e.g., 1 psi range)
            P_min = -1.0
            P_max = 1.0
            out_min = 1638.0 # 10% of 2^14
            out_max = 14745.0 # 90% of 2^14
            
            pressure_psi = ((dp_raw - out_min) * (P_max - P_min) / (out_max - out_min)) + P_min
            pressure_pa = pressure_psi * 6894.76 # Convert PSI to Pascals
            
            # Prevent negative square root if pressure fluctuates near zero
            if pressure_pa < 0:
                return 0.0
                
            # Calculate Airspeed (Bernoulli's equation): V = sqrt(2 * dp / rho)
            # Assuming standard air density (rho) at sea level = 1.225 kg/m^3
            air_density = 1.225
            airspeed = math.sqrt((2 * pressure_pa) / air_density)
            
            return airspeed
            
        except Exception as e:
            return None

    def close(self):
        if self.bus:
            self.bus.close()
