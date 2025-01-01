#include "dynamics.hpp"
#include <cmath>

Dynamics::Dynamics(std::shared_ptr<Rocket> rocket) : rocket(rocket) {}

Dynamics::State Dynamics::computeStateDerivatives(const State& state, double t) {
    State derivatives;
    const double g = 9.81; // m/s^2
    double burnTime = rocket->getBurnTime();
    double massFraction = (t < burnTime) ? 
        (rocket->getWetMass() - rocket->getDryMass()) * (burnTime - t) / burnTime : 0.0;
    double currentMass = rocket->getDryMass() + massFraction;
    double thrust = rocket->getThrust(t);
    Eigen::Matrix3d R = quaternionToRotationMatrix(state.quaternion);

    // Forces in body frame
    Eigen::Vector3d gravity = R * Eigen::Vector3d(0, 0, -currentMass * g);
    Eigen::Vector3d thrustForce(0.95 * thrust, 0, 0); // 95% thrust efficiency as per MATLAB

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            derivatives.position = R.transpose() * state.velocity;
        }

        #pragma omp section
        {
            Eigen::Vector3d totalForce = gravity + thrustForce;
            derivatives.velocity = totalForce / currentMass - 
                                 state.angularVel.cross(state.velocity);
        }

        #pragma omp section
        {
            Eigen::Vector4d qDot;
            qDot(0) = 0.5 * (-state.angularVel.x() * state.quaternion(1) -
                             state.angularVel.y() * state.quaternion(2) -
                             state.angularVel.z() * state.quaternion(3));
            qDot(1) = 0.5 * (state.angularVel.x() * state.quaternion(0) +
                             state.angularVel.z() * state.quaternion(2) -
                             state.angularVel.y() * state.quaternion(3));
            qDot(2) = 0.5 * (state.angularVel.y() * state.quaternion(0) -
                             state.angularVel.z() * state.quaternion(1) +
                             state.angularVel.x() * state.quaternion(3));
            qDot(3) = 0.5 * (state.angularVel.z() * state.quaternion(0) +
                             state.angularVel.y() * state.quaternion(1) -
                             state.angularVel.x() * state.quaternion(2));
            derivatives.quaternion = qDot;
        }
    }

    return derivatives;
}

Eigen::Matrix3d Dynamics::quaternionToRotationMatrix(const Eigen::Vector4d& q) {
    double q0 = q(0), q1 = q(1), q2 = q(2), q3 = q(3);
    Eigen::Matrix3d R;
    
    R << 1 - 2*(q2*q2 + q3*q3), 2*(q1*q2 - q0*q3), 2*(q1*q3 + q0*q2),
         2*(q1*q2 + q0*q3), 1 - 2*(q1*q1 + q3*q3), 2*(q2*q3 - q0*q1),
         2*(q1*q3 - q0*q2), 2*(q2*q3 + q0*q1), 1 - 2*(q1*q1 + q2*q2);
    
    return R;
}

Eigen::Vector4d Dynamics::integrateQuaternion(
    const Eigen::Vector4d& q, 
    const Eigen::Vector3d& w, 
    double dt
) {
    Eigen::Matrix4d omegaMatrix;
    omegaMatrix << 
        0.0,    -w.x(), -w.y(), -w.z(),
        w.x(),   0.0,    w.z(),  -w.y(),
        w.y(),  -w.z(),  0.0,    w.x(),
        w.z(),   w.y(),  -w.x(), 0.0;
    Eigen::Vector4d dqdt = 0.5 * omegaMatrix * q;
    Eigen::Vector4d nextQ = q + dt * dqdt;
    return nextQ.normalized();
}