// rocket.cpp
#include "rocket.hpp"
#include <stdexcept>
#include <algorithm>

Rocket::Rocket() {
    props.wetMass = 0.0;
    props.dryMass = 0.0;
    props.wetMOI = Eigen::Matrix3d::Zero();
    props.dryMOI = Eigen::Matrix3d::Zero();
    props.initialCG = 0.0;
    props.finalCG = 0.0;
    props.CP = 0.0;
    props.length = 0.0;
    props.diameter = 0.0;
    props.radius = 0.0;
    props.burnTime = 0.0;
}

void Rocket::setWetMass(double mass) {
    if (mass < 0) throw std::invalid_argument("Wet mass must be positive");
    props.wetMass = mass;
}

void Rocket::setBurnTime(double burnTime) {
    if (burnTime < 0) throw std::invalid_argument("Wet mass must be positive");
    props.burnTime = burnTime;
}

void Rocket::setDryMass(double mass) {
    if (mass < 0) throw std::invalid_argument("Dry mass must be positive");
    props.dryMass = mass;
}

void Rocket::setWetMOI(const Eigen::Matrix3d& moi) {
    if ((moi.diagonal().array() <= 0).any()) {
        throw std::invalid_argument("MOI diagonal elements must be positive");
    }
    props.wetMOI = moi;
}

void Rocket::setDryMOI(const Eigen::Matrix3d& moi) {
    if ((moi.diagonal().array() <= 0).any()) {
        throw std::invalid_argument("MOI diagonal elements must be positive");
    }
    props.dryMOI = moi;
}

void Rocket::setThrustCurve(const std::vector<std::pair<double, double>>& curve) {
    if (curve.empty()) 
        throw std::invalid_argument("Thrust curve cannot be empty");
    props.thrustCurve = curve;
}

double Rocket::getThrust(double t) const {
    if (t < 0 || props.thrustCurve.empty()) return 0.0;
    auto it = std::lower_bound(props.thrustCurve.begin(), props.thrustCurve.end(), 
                             std::make_pair(t, 0.0),
                             [](const auto& a, const auto& b) { return a.first < b.first; });
    
    if (it == props.thrustCurve.begin()) return it->second;
    if (it == props.thrustCurve.end()) return props.thrustCurve.back().second;
    
    auto prev = std::prev(it);
    double t1 = prev->first;
    double t2 = it->first;
    double f1 = prev->second;
    double f2 = it->second;
    
    return f1 + (f2 - f1) * (t - t1) / (t2 - t1);
}

const std::vector<std::pair<double, double>>& Rocket::getThrustCurve() const {
    return props.thrustCurve;
}

double Rocket::getWetMass() const {
    return props.wetMass;
}

double Rocket::getDryMass() const {
    return props.dryMass;
}

const Eigen::Matrix3d& Rocket::getWetMOI() const {
    return props.wetMOI;
}

const Eigen::Matrix3d& Rocket::getDryMOI() const {
    return props.dryMOI;
}

double Rocket::getInitialCG() const {
    return props.initialCG;
}

double Rocket::getFinalCG() const {
    return props.finalCG;
}

double Rocket::getCP() const {
    return props.CP;
}

double Rocket::getLength() const {
    return props.length;
}

double Rocket::getDiameter() const {
    return props.diameter;
}

double Rocket::getBurnTime() const {
    return props.burnTime;
}
