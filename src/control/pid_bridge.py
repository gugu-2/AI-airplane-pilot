import ctypes
import os

class CppPIDController:
    """
    Python wrapper for the high-performance C++ PID Controller.
    Used for sub-millisecond latency actuator commands.
    """
    def __init__(self, kp: float, ki: float, kd: float, min_out: float, max_out: float):
        # Locate the compiled shared library (assuming it's built in a 'build' folder)
        # For simulation without a compiler, we will mock the DLL load failure gracefully.
        lib_path = os.path.join(os.path.dirname(__file__), 'build', 'pid_controller.so')
        
        # Windows uses .dll, Linux uses .so
        if os.name == 'nt':
            lib_path = os.path.join(os.path.dirname(__file__), 'build', 'pid_controller.dll')
            
        self._lib_loaded = False
        if os.path.exists(lib_path):
            try:
                self.lib = ctypes.CDLL(lib_path)
                
                # Define argtypes and restypes for C-wrapper
                self.lib.PIDController_new.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double]
                self.lib.PIDController_new.restype = ctypes.c_void_p
                
                self.lib.PIDController_compute.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_double, ctypes.c_double]
                self.lib.PIDController_compute.restype = ctypes.c_double
                
                self.lib.PIDController_delete.argtypes = [ctypes.c_void_p]
                
                # Initialize C++ object
                self.obj = self.lib.PIDController_new(kp, ki, kd, min_out, max_out)
                self._lib_loaded = True
            except Exception as e:
                print(f"[C++ Bridge] Failed to load C++ PID library: {e}")
        else:
            print("[C++ Bridge] WARNING: C++ PID library not compiled. Falling back to Python simulation mock.")
            self._mock_kp = kp
            self._mock_ki = ki
            self._mock_kd = kd
            self._mock_min = min_out
            self._mock_max = max_out
            self._mock_integral = 0.0
            self._mock_prev_error = 0.0
            
    def compute(self, setpoint: float, measurement: float, dt: float) -> float:
        if self._lib_loaded:
            return self.lib.PIDController_compute(self.obj, setpoint, measurement, dt)
        else:
            # Python fallback if C++ library is missing (for local testing on Windows)
            error = setpoint - measurement
            self._mock_integral += error * dt
            derivative = (error - self._mock_prev_error) / dt if dt > 0 else 0
            
            output = (self._mock_kp * error) + (self._mock_ki * self._mock_integral) + (self._mock_kd * derivative)
            output = max(self._mock_min, min(self._mock_max, output))
            
            self._mock_prev_error = error
            return output

    def __del__(self):
        if self._lib_loaded:
            self.lib.PIDController_delete(self.obj)

# Quick test if run directly
if __name__ == "__main__":
    print("Testing Flight Control Pitch PID:")
    # kp=1.5, ki=0.1, kd=0.5, max angles = -45 to 45 degrees
    pitch_pid = CppPIDController(1.5, 0.1, 0.5, -45.0, 45.0)
    
    # Simulate climbing to 15 degrees pitch
    current_pitch = 0.0
    for step in range(5):
        command = pitch_pid.compute(setpoint=15.0, measurement=current_pitch, dt=0.1)
        print(f"Step {step}: Current Pitch: {current_pitch:.2f} | PID outputs actuator command: {command:.2f}")
        current_pitch += command * 0.2 # simulate drone reacting
