try:
    from smbus2 import SMBus
    SMBUS_AVAILABLE = True
except ImportError:
    SMBUS_AVAILABLE = False
import time
import math

class IMUDriver:
    """
    Embedded I2C driver for standard IMUs like MPU6050.
    Provides real-time 3-axis accelerometer and gyroscope data.
    """
    def __init__(self, bus_num=1, address=0x68):
        self.bus_num = bus_num
        self.address = address
        self.bus = None
        self.initialized = False
        
        if not SMBUS_AVAILABLE:
            raise ImportError("smbus2 library not available. Hardware IMU cannot be initialized.")
            
        try:
            self.bus = SMBus(self.bus_num)
            # Wake up the MPU6050 (write 0 to power management register 0x6B)
            self.bus.write_byte_data(self.address, 0x6B, 0)
            self.initialized = True
        except Exception as e:
            raise ConnectionError(f"Failed to connect to IMU at I2C bus {bus_num}, address {hex(address)}: {e}")

    def _read_word_2c(self, reg):
        """Reads two bytes from an I2C register and converts to 2's complement."""
        high = self.bus.read_byte_data(self.address, reg)
        low = self.bus.read_byte_data(self.address, reg+1)
        val = (high << 8) + low
        if val >= 0x8000:
            return -((65535 - val) + 1)
        else:
            return val

    def read_acceleration(self):
        """Returns acceleration in m/s^2 (Assuming +/- 2g range)"""
        if not self.initialized:
            return 0.0, 0.0, 0.0
            
        # Accelerometer registers start at 0x3B
        accel_x = self._read_word_2c(0x3B) / 16384.0 * 9.81
        accel_y = self._read_word_2c(0x3D) / 16384.0 * 9.81
        accel_z = self._read_word_2c(0x3F) / 16384.0 * 9.81
        
        return accel_x, accel_y, accel_z
        
    def read_gyroscope(self):
        """Returns rotational velocity in degrees/s"""
        if not self.initialized:
            return 0.0, 0.0, 0.0
            
        # Gyro registers start at 0x43
        gyro_x = self._read_word_2c(0x43) / 131.0
        gyro_y = self._read_word_2c(0x45) / 131.0
        gyro_z = self._read_word_2c(0x47) / 131.0
        
        return gyro_x, gyro_y, gyro_z

    def close(self):
        if self.bus:
            self.bus.close()
