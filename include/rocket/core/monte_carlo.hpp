// monte_carlo.hpp
#pragma once
#include <vector>
#include <random>
#include <omp.h>
#include "rocket.hpp"
#include "dynamics.hpp"
#include "environment.hpp"

class MonteCarlo {
public:
    struct Parameters {
        struct {
            double wetMass_mean, wetMass_3sigma;
            double dryMass_mean, dryMass_3sigma;
            double burnTime_mean, burnTime_3sigma;
            double thrust_3sigma;
            Eigen::Vector3d wetMOI_3sigma;
            Eigen::Vector3d dryMOI_3sigma;
        } rocketParams;
        struct {
            double gustSpeed_mean, gustSpeed_3sigma;
            double gustLength_mean, gustLength_3sigma;
        } environmentParams;
    };

    struct SimulationResult {
        std::vector<Dynamics::State> trajectory;
        double maxAltitude;
        double maxVelocity;
        double flightTime;
    };
    MonteCarlo(const Parameters& params);
    std::vector<SimulationResult> runSimulations(int numRuns);

private:
    Parameters params;
    std::mt19937 rng;
    double generateGaussianRandom(double mean, double sigma);
    std::shared_ptr<Rocket> generateRandomRocket();
    Environment::WindParameters generateRandomWind();
};