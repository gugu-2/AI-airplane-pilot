import numpy as np

class ExtendedKalmanFilter:
    """
    A foundational Extended Kalman Filter (EKF) for Sensor Fusion.
    Fuses high-frequency, noisy IMU data (acceleration) with low-frequency, 
    noisy GPS data (position) to produce a highly accurate state estimation.
    """
    def __init__(self, initial_x=0.0, initial_v=0.0):
        # State vector: [Position, Velocity]
        self.X = np.array([[initial_x], [initial_v]])
        
        # Initial Uncertainty (Covariance) Matrix
        self.P = np.array([[1.0, 0.0],
                           [0.0, 1.0]])
        
        # Process Noise Covariance (Uncertainty in the model/IMU)
        # Assumes IMU has some variance
        self.Q = np.array([[0.01, 0.0],
                           [0.0, 0.01]])
        
        # Measurement Noise Covariance (Uncertainty in GPS)
        # GPS is typically less accurate than IMU over short intervals
        self.R = np.array([[0.5]])

        print("[Sensor Fusion] Extended Kalman Filter initialized.")

    def predict(self, accel_measurement, dt):
        """
        Step 1: Predict the next state using physics (IMU acceleration data).
        Kinematics: 
        pos = pos + vel*dt + 0.5*accel*dt^2
        vel = vel + accel*dt
        """
        # State Transition Matrix
        F = np.array([[1.0, dt],
                      [0.0, 1.0]])
                      
        # Control Input Matrix
        B = np.array([[0.5 * dt**2],
                      [dt]])
                      
        # Control Input (IMU Acceleration)
        u = np.array([[accel_measurement]])

        # Predict State Forward
        self.X = np.dot(F, self.X) + np.dot(B, u)
        
        # Predict Uncertainty Forward
        self.P = np.dot(np.dot(F, self.P), F.T) + self.Q

        return self.X[0][0], self.X[1][0]

    def update(self, gps_measurement):
        """
        Step 2: Update the state using an absolute measurement (GPS).
        """
        # Observation Matrix (We only measure position from GPS, not velocity)
        H = np.array([[1.0, 0.0]])
        
        # Calculate Measurement Residual (Difference between predicted pos and actual GPS)
        Z = np.array([[gps_measurement]])
        Y = Z - np.dot(H, self.X)
        
        # Calculate Kalman Gain (How much should we trust the GPS vs our Prediction?)
        S = np.dot(np.dot(H, self.P), H.T) + self.R
        K = np.dot(np.dot(self.P, H.T), np.linalg.inv(S))
        
        # Update State with the Gain
        self.X = self.X + np.dot(K, Y)
        
        # Update Uncertainty Matrix
        I = np.eye(self.P.shape[0])
        self.P = np.dot((I - np.dot(K, H)), self.P)

        return self.X[0][0], self.X[1][0]

if __name__ == "__main__":
    print("Testing Sensor Fusion (IMU + GPS) via EKF:")
    ekf = ExtendedKalmanFilter(initial_x=0.0, initial_v=0.0)
    
    # Simulated true state
    true_pos = 0.0
    true_vel = 10.0
    
    for step in range(1, 11):
        # 1. IMU predicts state at 100Hz (dt = 0.01)
        # True accel is 0, but IMU has noise
        noisy_accel = np.random.normal(0, 0.1) 
        pred_pos, pred_vel = ekf.predict(accel_measurement=noisy_accel, dt=0.01)
        
        # Update true physics for simulation
        true_pos += true_vel * 0.01
        
        # 2. GPS updates state at 10Hz (every 10 IMU ticks)
        # GPS has high noise
        noisy_gps = true_pos + np.random.normal(0, 2.0)
        
        final_pos, final_vel = ekf.update(gps_measurement=noisy_gps)
        
        print(f"Step {step}:")
        print(f"  True Pos: {true_pos:.3f}m | Noisy GPS: {noisy_gps:.3f}m")
        print(f"  EKF Fused Pos: {final_pos:.3f}m | EKF Fused Vel: {final_vel:.3f}m/s")
        print("-" * 50)
