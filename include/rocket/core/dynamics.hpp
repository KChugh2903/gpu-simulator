// dynamics.hpp
#pragma once
#include <Eigen/Dense>
#include "rocket.hpp"
#include <omp.h>

class Dynamics {
public:
    struct State {
        Eigen::Vector3d position;   
        Eigen::Vector3d velocity;   
        Eigen::Vector4d quaternion; 
        Eigen::Vector3d angularVel;  
    };

    // Constructor
    Dynamics(std::shared_ptr<Rocket> rocket);
    State computeStateDerivatives(const State& state, double t);
    State integrate(const State& state, double dt);

private:
    std::shared_ptr<Rocket> rocket;
    Eigen::Matrix3d quaternionToRotationMatrix(const Eigen::Vector4d& q);
    Eigen::Vector4d integrateQuaternion(const Eigen::Vector4d& q, const Eigen::Vector3d& w, double dt);
};
