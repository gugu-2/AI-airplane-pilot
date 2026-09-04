#ifndef PID_CONTROLLER_H
#define PID_CONTROLLER_H

#include <chrono>

class PIDController {
public:
    PIDController(double kp, double ki, double kd, double min_out, double max_out);
    
    // Calculates the required control output (e.g., servo PWM or angle)
    // based on the setpoint (desired state) and current measurement.
    double compute(double setpoint, double measurement, double dt);
    
    // Resets the integral and derivative terms
    void reset();

    // Dynamically update gains (useful for tuning during flight)
    void set_gains(double kp, double ki, double kd);

private:
    double _kp, _ki, _kd;
    double _min_out, _max_out;
    
    double _integral;
    double _prev_error;
};

#endif // PID_CONTROLLER_H
