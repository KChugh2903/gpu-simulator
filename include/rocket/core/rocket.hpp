// rocket.hpp
#pragma once
#include <Eigen/Dense>
#include <vector>
#include <memory>

class Rocket {
public:
    Rocket();
    ~Rocket() = default;
    struct Properties {
        double wetMass;      // kg
        double dryMass;      // kg
        Eigen::Matrix3d wetMOI;  // kg*m^2
        Eigen::Matrix3d dryMOI;  // kg*m^2
        double initialCG;    // m from nosecone
        double finalCG;      // m from nosecone
        double CP;           // m from nosecone
        double length;       // m
        double diameter;     // m
        double radius;       // m
        double burnTime;     // s
        std::vector<std::pair<double, double>> thrustCurve; 
    };
    void setWetMass(double mass);
    void setDryMass(double mass);
    void setWetMOI(const Eigen::Matrix3d& moi);
    void setDryMOI(const Eigen::Matrix3d& moi);
    void setInitialCG(double cg);
    void setFinalCG(double cg);
    void setCP(double cp);
    void setLength(double len);
    void setDiameter(double dia);
    void setBurnTime(double time);
    void setThrustCurve(const std::vector<std::pair<double, double>>& curve);

    // Getters
    double getWetMass() const;
    double getDryMass() const;
    const Eigen::Matrix3d& getWetMOI() const;
    const Eigen::Matrix3d& getDryMOI() const;
    double getInitialCG() const;
    double getFinalCG() const;
    double getCP() const;
    double getLength() const;
    double getDiameter() const;
    double getBurnTime() const;
    double getThrust(double t) const;
    const std::vector<std::pair<double, double>>& getThrustCurve() const;

    
private:
    Properties props;
};