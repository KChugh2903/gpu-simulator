#include "monte_carlo.hpp"
#include <chrono>
#include <iostream>

MonteCarlo::MonteCarlo(const Parameters& params) 
    : params(params), 
      rng(std::chrono::system_clock::now().time_since_epoch().count()) {
}

std::vector<MonteCarlo::SimulationResult> MonteCarlo::runSimulations(int numRuns) {
    std::vector<SimulationResult> results(numRuns);
    int progressStep = std::max(1, numRuns / 100);
    auto startTime = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < numRuns; ++i) {
        try {
            auto rocket = generateRandomRocket();
            auto windParams = generateRandomWind();
            Environment env(0.0, 0.0, 0.0); // Default location
            env.addWindParameter(windParams.gustSpeed,
                               windParams.gustLength, params.environmentParams.gustLength_3sigma);
            Dynamics dynamics(rocket);
            Dynamics::State state;
            state.position = Eigen::Vector3d(0, 0, 0);
            state.velocity = Eigen::Vector3d(0.001, 0, 0);  // Small initial velocity
            state.quaternion = Eigen::Vector4d(1, 0, 0, 0); // Initially vertical
            const double dt = 0.001;  // Time step
            const double maxTime = rocket->getBurnTime() * 1.5;  // Simulate past burnout
            std::vector<Dynamics::State> trajectory;
            double maxAlt = 0.0;
            double maxVel = 0.0;
            for (double t = 0; t < maxTime; t += dt) {
                trajectory.push_back(state);
                maxAlt = std::max(maxAlt, state.position.z());
                maxVel = std::max(maxVel, state.velocity.norm());
                if (state.position.z() < 0 && t > 1.0) break;
                state = dynamics.integrate(state, dt);
            }
            #pragma omp critical
            {
                results[i].trajectory = std::move(trajectory);
                results[i].maxAltitude = maxAlt;
                results[i].maxVelocity = maxVel;
                results[i].flightTime = trajectory.size() * dt;
                if (i % progressStep == 0) {
                    auto now = std::chrono::high_resolution_clock::now();
                    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - startTime).count();
                    std::cout << "Progress: " << (i * 100.0 / numRuns) 
                              << "% (" << elapsed << "s)\r" << std::flush;
                }
            }
        } catch (const std::exception& e) {
            #pragma omp critical
            {
                std::cerr << "Simulation " << i << " failed: " << e.what() << std::endl;
            }
        }
    }
    
    std::cout << "\nSimulations completed." << std::endl;
    return results;
}

double MonteCarlo::generateGaussianRandom(double mean, double sigma) {
    std::normal_distribution<double> dist(mean, sigma/3.0);
    return dist(rng);
}

std::shared_ptr<Rocket> MonteCarlo::generateRandomRocket() {
    auto rocket = std::make_shared<Rocket>();
    double wetMass = generateGaussianRandom(
        params.rocketParams.wetMass_mean,
        params.rocketParams.wetMass_3sigma);
    
    double dryMass = generateGaussianRandom(
        params.rocketParams.dryMass_mean,
        params.rocketParams.dryMass_3sigma);
    
    double burnTime = generateGaussianRandom(
        params.rocketParams.burnTime_mean,
        params.rocketParams.burnTime_3sigma);
    
    if (wetMass <= dryMass) {
        wetMass = dryMass * 1.1;
    }
    
    rocket->setWetMass(wetMass);
    rocket->setDryMass(dryMass);
    rocket->setBurnTime(burnTime);
    Eigen::Matrix3d wetMOI = rocket->getWetMOI();
    Eigen::Matrix3d dryMOI = rocket->getDryMOI();
    for (int i = 0; i < 3; i++) {
        wetMOI(i,i) *= (1.0 + generateGaussianRandom(0, params.rocketParams.wetMOI_3sigma[i]));
        dryMOI(i,i) *= (1.0 + generateGaussianRandom(0, params.rocketParams.dryMOI_3sigma[i]));
    }
    rocket->setWetMOI(wetMOI);
    rocket->setDryMOI(dryMOI);
    auto baseThrustCurve = rocket->getThrustCurve();
    std::vector<std::pair<double, double>> variedThrustCurve;
    variedThrustCurve.reserve(baseThrustCurve.size());
    
    double thrustVariation = 1.0 + generateGaussianRandom(0, params.rocketParams.thrust_3sigma);
    for (const auto& point : baseThrustCurve) {
        variedThrustCurve.emplace_back(point.first, point.second * thrustVariation);
    }
    rocket->setThrustCurve(variedThrustCurve);
    return rocket;
}

Environment::WindParameters MonteCarlo::generateRandomWind() {
    Environment::WindParameters wind;
    wind.gustSpeed = generateGaussianRandom(
        params.environmentParams.gustSpeed_mean,
        params.environmentParams.gustSpeed_3sigma);
    
    wind.gustLength = generateGaussianRandom(
        params.environmentParams.gustLength_mean,
        params.environmentParams.gustLength_3sigma);
    std::uniform_real_distribution<double> angleDist(0, 2*M_PI);
    double azimuth = angleDist(rng);
    double elevation = std::acos(std::uniform_real_distribution<double>(-1, 1)(rng));
    wind.direction[0] = std::sin(elevation) * std::cos(azimuth);
    wind.direction[1] = std::sin(elevation) * std::sin(azimuth);
    wind.direction[2] = std::cos(elevation);
    return wind;
}

Dynamics::State Dynamics::integrate(const State& state, double dt) {
    // RK4 integration method
    State k1 = computeStateDerivatives(state, 0.0);
    
    State k1_intermediate;
    k1_intermediate.position = state.position + 0.5 * dt * k1.position;
    k1_intermediate.velocity = state.velocity + 0.5 * dt * k1.velocity;
    k1_intermediate.quaternion = integrateQuaternion(state.quaternion, state.angularVel + 0.5 * dt * k1.angularVel, 0.5 * dt);
    k1_intermediate.angularVel = state.angularVel + 0.5 * dt * k1.angularVel;
    State k2 = computeStateDerivatives(k1_intermediate, 0.5 * dt);
    
    State k2_intermediate;
    k2_intermediate.position = state.position + 0.5 * dt * k2.position;
    k2_intermediate.velocity = state.velocity + 0.5 * dt * k2.velocity;
    k2_intermediate.quaternion = integrateQuaternion(state.quaternion, state.angularVel + 0.5 * dt * k2.angularVel, 0.5 * dt);
    k2_intermediate.angularVel = state.angularVel + 0.5 * dt * k2.angularVel;
    State k3 = computeStateDerivatives(k2_intermediate, 0.5 * dt);
    
    State k4_intermediate;
    k4_intermediate.position = state.position + dt * k3.position;
    k4_intermediate.velocity = state.velocity + dt * k3.velocity;
    k4_intermediate.quaternion = integrateQuaternion(state.quaternion, state.angularVel + dt * k3.angularVel, dt);
    k4_intermediate.angularVel = state.angularVel + dt * k3.angularVel;
    State k4 = computeStateDerivatives(k4_intermediate, dt);
    
    // Combine derivatives with weighted average
    State nextState;
    nextState.position = state.position + (dt / 6.0) * (k1.position + 2 * k2.position + 2 * k3.position + k4.position);
    nextState.velocity = state.velocity + (dt / 6.0) * (k1.velocity + 2 * k2.velocity + 2 * k3.velocity + k4.velocity);
    nextState.angularVel = state.angularVel + (dt / 6.0) * (k1.angularVel + 2 * k2.angularVel + 2 * k3.angularVel + k4.angularVel);
    
    // Special handling for quaternion integration
    Eigen::Vector4d q1 = integrateQuaternion(state.quaternion, state.angularVel, dt/6.0);
    Eigen::Vector4d q2 = integrateQuaternion(state.quaternion, state.angularVel + 2 * k2.angularVel, dt/6.0);
    Eigen::Vector4d q3 = integrateQuaternion(state.quaternion, state.angularVel + 2 * k3.angularVel, dt/6.0);
    Eigen::Vector4d q4 = integrateQuaternion(state.quaternion, state.angularVel + k4.angularVel, dt/6.0);
    
    // Weighted average of quaternions with normalization
    nextState.quaternion = (q1 + 2 * q2 + 2 * q3 + q4).normalized();
    
    return nextState;
}