#include <cmath>
#include <iostream>

// Export C-compatible functions so Python's ctypes can call them
extern "C" {

    /**
     * Structure representing the EKF State memory block.
     * Passed back and forth between Python and C++ as raw pointers.
     */
    struct EKFState {
        double lat;
        double lon;
        double v_lat;
        double v_lon;
        double P[4][4]; // 4x4 Covariance Matrix
    };

    /**
     * Initializes the EKF State block in memory.
     */
    void init_ekf(EKFState* state, double initial_lat, double initial_lon) {
        state->lat = initial_lat;
        state->lon = initial_lon;
        state->v_lat = 0.0;
        state->v_lon = 0.0;
        
        // Initialize Covariance Matrix (Identity * 1.0)
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                state->P[i][j] = (i == j) ? 1.0 : 0.0;
            }
        }
    }

    /**
     * Fast C++ EKF Predict Phase (IMU Physics)
     */
    void predict_ekf(EKFState* state, double dt, double accel_lat, double accel_lon) {
        // 1. Predict State: x = F*x + B*u
        // Because F is very sparse, we can optimize the matrix multiplication into basic algebra
        
        // New velocities
        double new_v_lat = state->v_lat + (accel_lat * dt);
        double new_v_lon = state->v_lon + (accel_lon * dt);
        
        // New positions
        state->lat = state->lat + (state->v_lat * dt) + (0.5 * accel_lat * dt * dt);
        state->lon = state->lon + (state->v_lon * dt) + (0.5 * accel_lon * dt * dt);
        
        state->v_lat = new_v_lat;
        state->v_lon = new_v_lon;
        
        // 2. Predict Covariance: P = F*P*F^T + Q
        // (Simplified for performance, assuming independent noise)
        double Q = 0.001; // Process noise
        
        state->P[0][0] += (dt * dt * state->P[2][2]) + Q;
        state->P[1][1] += (dt * dt * state->P[3][3]) + Q;
        state->P[2][2] += Q;
        state->P[3][3] += Q;
    }

    /**
     * Fast C++ EKF Update Phase (GPS Fusion)
     */
    void update_ekf(EKFState* state, double gps_lat, double gps_lon) {
        double R = 0.05; // Measurement Noise
        
        // 1. Calculate Innovation Residual (y)
        double y_lat = gps_lat - state->lat;
        double y_lon = gps_lon - state->lon;
        
        // 2. Innovation Covariance (S)
        double S_lat = state->P[0][0] + R;
        double S_lon = state->P[1][1] + R;
        
        // 3. Kalman Gain (K = P / S)
        double K_lat = state->P[0][0] / S_lat;
        double K_lon = state->P[1][1] / S_lon;
        
        // 4. Update State
        state->lat = state->lat + (K_lat * y_lat);
        state->lon = state->lon + (K_lon * y_lon);
        
        // 5. Update Covariance (P = (1 - K) * P)
        state->P[0][0] = (1.0 - K_lat) * state->P[0][0];
        state->P[1][1] = (1.0 - K_lon) * state->P[1][1];
    }
}
