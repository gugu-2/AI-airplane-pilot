#include "pid_controller.h"
#include <algorithm>
#include <iostream>

PIDController::PIDController(double kp, double ki, double kd, double min_out, double max_out)
    : _kp(kp), _ki(ki), _kd(kd), _min_out(min_out), _max_out(max_out), _integral(0.0), _prev_error(0.0) {
    std::cout << "[Flight Control] C++ PID Initialized (Kp: " << _kp << ", Ki: " << _ki << ", Kd: " << _kd << ")\n";
}

double PIDController::compute(double setpoint, double measurement, double dt) {
    if (dt <= 0.0) {
        return 0.0;
    }

    // 1. Calculate error
    double error = setpoint - measurement;

    // 2. Proportional term
    double p_out = _kp * error;

    // 3. Integral term (with anti-windup clamping)
    _integral += error * dt;
    double i_out = _ki * _integral;
    
    // Anti-windup: restrict integral buildup if we are maxed out
    if (i_out > _max_out) i_out = _max_out;
    else if (i_out < _min_out) i_out = _min_out;

    // 4. Derivative term (prevent derivative kick by using measurement derivative if preferred, 
    // but standard error derivative is used here for simplicity)
    double derivative = (error - _prev_error) / dt;
    double d_out = _kd * derivative;

    // 5. Total Output
    double output = p_out + i_out + d_out;

    // 6. Clamp output to hardware limits (e.g., servo max angles)
    output = std::max(_min_out, std::min(_max_out, output));

    // Save error for next iteration
    _prev_error = error;

    return output;
}

void PIDController::reset() {
    _integral = 0.0;
    _prev_error = 0.0;
}

void PIDController::set_gains(double kp, double ki, double kd) {
    _kp = kp;
    _ki = ki;
    _kd = kd;
}

// C-wrapper for calling from Python via ctypes
extern "C" {
    PIDController* PIDController_new(double kp, double ki, double kd, double min_out, double max_out) {
        return new PIDController(kp, ki, kd, min_out, max_out);
    }
    
    double PIDController_compute(PIDController* pid, double setpoint, double measurement, double dt) {
        return pid->compute(setpoint, measurement, dt);
    }
    
    void PIDController_reset(PIDController* pid) {
        pid->reset();
    }
    
    void PIDController_delete(PIDController* pid) {
        delete pid;
    }
}
