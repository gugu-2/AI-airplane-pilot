import numpy as np
import ctypes
import os
import sys

# Try to load the C++ High-Performance Engine
cpp_engine = None
try:
    lib_ext = '.dll' if sys.platform == 'win32' else '.so'
    lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'cpp_core', f'fast_ekf{lib_ext}'))
    if os.path.exists(lib_path):
        cpp_engine = ctypes.CDLL(lib_path)
        print(f"[EKF] SUCCESS: Hardware-Accelerated C++ Engine loaded! ({lib_path})")
    else:
        print("[EKF] WARNING: C++ Engine not compiled. Falling back to Numpy Python math.")
except Exception as e:
    print(f"[EKF] WARNING: Failed to load C++ Engine ({e}). Falling back to Numpy.")

# Define the C-compatible structure for the EKF State
class CEKFState(ctypes.Structure):
    _fields_ = [
        ("lat", ctypes.c_double),
        ("lon", ctypes.c_double),
        ("v_lat", ctypes.c_double),
        ("v_lon", ctypes.c_double),
        ("P", (ctypes.c_double * 4) * 4) # 4x4 array
    ]

class ExtendedKalmanFilter:
    """
    A simplified 2D Extended Kalman Filter for drone state estimation.
    Tracks: [lat, lon, alt, v_lat, v_lon, v_alt]
    B1 FIX: Altitude is now fused into the EKF state vector.
    Uses C++ engine if compiled, otherwise falls back to Numpy.
    """
    def __init__(self, initial_lat, initial_lon, initial_alt=0.0):
        self.use_cpp = cpp_engine is not None
        
        if self.use_cpp:
            cpp_engine.init_ekf.argtypes = [ctypes.POINTER(CEKFState), ctypes.c_double, ctypes.c_double]
            cpp_engine.predict_ekf.argtypes = [ctypes.POINTER(CEKFState), ctypes.c_double, ctypes.c_double, ctypes.c_double]
            cpp_engine.update_ekf.argtypes = [ctypes.POINTER(CEKFState), ctypes.c_double, ctypes.c_double]
            self.c_state = CEKFState()
            cpp_engine.init_ekf(ctypes.byref(self.c_state), initial_lat, initial_lon)
            # B1: Track altitude separately in Numpy even in C++ mode (C++ backend is 2D only)
            self._alt = initial_alt
            self._v_alt = 0.0
        else:
            # B1 FIX: Extended to 6-dim state: [lat, lon, alt, v_lat, v_lon, v_alt]
            self.x = np.array([initial_lat, initial_lon, initial_alt, 0.0, 0.0, 0.0], dtype=float)
            self.P = np.eye(6) * 1.0
            self.Q = np.eye(6) * 0.001
            self.R = np.eye(3) * 0.05   # GPS noise: lat, lon, alt
            self.H = np.zeros((3, 6))
            self.H[0, 0] = 1.0  # lat
            self.H[1, 1] = 1.0  # lon
            self.H[2, 2] = 1.0  # alt

    def predict(self, dt: float, accel_lat: float, accel_lon: float, accel_alt: float = 0.0):
        if self.use_cpp:
            cpp_engine.predict_ekf(ctypes.byref(self.c_state), dt, accel_lat, accel_lon)
            # B1: Update altitude state manually
            self._alt += self._v_alt * dt + 0.5 * accel_alt * dt**2
            self._v_alt += accel_alt * dt
        else:
            # B1 FIX: 6-dim state transition matrix
            F = np.zeros((6, 6))
            for i in range(6):
                F[i, i] = 1.0
            F[0, 3] = dt  # lat += v_lat * dt
            F[1, 4] = dt  # lon += v_lon * dt
            F[2, 5] = dt  # alt += v_alt * dt
            B = np.array([
                [0.5*dt**2, 0, 0],
                [0, 0.5*dt**2, 0],
                [0, 0, 0.5*dt**2],
                [dt, 0, 0],
                [0, dt, 0],
                [0, 0, dt]
            ])
            u = np.array([accel_lat, accel_lon, accel_alt])
            self.x = np.dot(F, self.x) + np.dot(B, u)
            self.P = np.dot(np.dot(F, self.P), F.T) + self.Q

    def update(self, gps_lat: float, gps_lon: float, gps_alt: float = None):
        if self.use_cpp:
            cpp_engine.update_ekf(ctypes.byref(self.c_state), gps_lat, gps_lon)
            # B1: Fuse GPS altitude with simple complementary filter
            if gps_alt is not None:
                self._alt = 0.8 * self._alt + 0.2 * gps_alt
            return self.c_state.lat, self.c_state.lon, self._alt
        else:
            # B1 FIX: Include altitude in GPS measurement if provided
            if gps_alt is not None:
                z = np.array([gps_lat, gps_lon, gps_alt])
                H = self.H  # 3x6
                R = self.R  # 3x3
            else:
                z = np.array([gps_lat, gps_lon])
                H = self.H[:2, :]  # 2x6 (lat/lon only)
                R = self.R[:2, :2]
            y = z - np.dot(H, self.x)
            S = np.dot(np.dot(H, self.P), H.T) + R
            K = np.dot(np.dot(self.P, H.T), np.linalg.inv(S))
            self.x = self.x + np.dot(K, y)
            I = np.eye(self.P.shape[0])
            self.P = np.dot((I - np.dot(K, H)), self.P)
            return self.x[0], self.x[1], self.x[2]

    def get_state(self):
        """Returns (lat, lon) for backward compatibility."""
        if self.use_cpp:
            return self.c_state.lat, self.c_state.lon
        return self.x[0], self.x[1]

    def get_state_3d(self):
        """Returns (lat, lon, alt) for 3D navigation/landing."""
        if self.use_cpp:
            return self.c_state.lat, self.c_state.lon, self._alt
        return self.x[0], self.x[1], self.x[2]

    def inject_position_delta(self, lat_shift: float, lon_shift: float, alt_shift: float = 0.0):
        """
        Safely injects a position delta into the EKF state.
        B1 FIX: Now also accepts alt_shift for vertical dead reckoning.
        """
        if self.use_cpp:
            self.c_state.lat += lat_shift
            self.c_state.lon += lon_shift
            self._alt += alt_shift
        else:
            self.x[0] += lat_shift
            self.x[1] += lon_shift
            self.x[2] += alt_shift

