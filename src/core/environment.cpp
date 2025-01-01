
// environment.cpp
#include "environment.hpp"
#include <algorithm>
#include <stdexcept>
#include <cmath>
#include <fstream>
#include <sstream>
#include <iostream>

Environment::Environment(double lat, double lon, double elev, bool useDefaultData)
    : latitude(lat), longitude(lon), elevation(elev), currentWindIndex(0) {
    initializeAtmosphericData(useDefaultData);
}

void Environment::initializeAtmosphericData(bool useDefaultData) {
    if (useDefaultData) {
        loadDefaultData();
    } else {
        generateEnvironmentData();
    }
}

void Environment::loadDefaultData() {
    // Load data from the default CSV file
    std::ifstream file("EnvironmentData_default.csv");
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open default environment data file");
    }

    // Skip header
    std::string line;
    std::getline(file, line);

    // Read data
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        AtmosphericData data;
        std::string value;

        // Read each column
        std::getline(ss, value, ',');
        data.altitude = std::stod(value);
        std::getline(ss, value, ',');
        data.density = std::stod(value);
        std::getline(ss, value, ',');
        data.pressure = std::stod(value);
        std::getline(ss, value, ',');
        data.temperature = std::stod(value);
        std::getline(ss, value, ',');
        data.speedOfSound = std::stod(value);
        std::getline(ss, value, ',');
        data.windSpeed = std::stod(value);

        atmosphericTable.push_back(data);
    }
}

void Environment::generateEnvironmentData() {
    // Generate atmospheric data using standard atmosphere model
    const int numPoints = 2000;
    const double maxAltitude = 20000.0;

    for (int i = 0; i < numPoints; ++i) {
        AtmosphericData data;
        data.altitude = i * maxAltitude / numPoints;

        // Standard atmosphere calculations
        double h = data.altitude / 1000.0; // km
        if (h <= 11.0) {
            // Troposphere
            data.temperature = 288.15 - 6.5 * h;
            data.pressure = 101325.0 * std::pow(288.15 / (288.15 - 6.5 * h), -5.255877);
        } else {
            // Stratosphere
            data.temperature = 216.65;
            data.pressure = 22632.1 * std::exp(-0.1577 * (h - 11.0));
        }

        // Derived quantities
        data.density = data.pressure / (287.05 * data.temperature);
        data.speedOfSound = std::sqrt(1.4 * 287.05 * data.temperature);
        data.windSpeed = 0.0; // Base wind speed, modified by wind parameters

        atmosphericTable.push_back(data);
    }
}

double Environment::interpolateData(double altitude, DataType type) const {
    auto it = std::lower_bound(atmosphericTable.begin(), atmosphericTable.end(), altitude,
        [](const AtmosphericData& data, double alt) {
            return data.altitude < alt;
        });

    if (it == atmosphericTable.begin()) {
        const auto& data = atmosphericTable.front();
        switch(type) {
            case DataType::DENSITY: return data.density;
            case DataType::PRESSURE: return data.pressure;
            case DataType::TEMPERATURE: return data.temperature;
            case DataType::SPEED_OF_SOUND: return data.speedOfSound;
            case DataType::WIND_SPEED: return data.windSpeed;
        }
    }
    
    if (it == atmosphericTable.end()) {
        const auto& data = atmosphericTable.back();
        switch(type) {
            case DataType::DENSITY: return data.density;
            case DataType::PRESSURE: return data.pressure;
            case DataType::TEMPERATURE: return data.temperature;
            case DataType::SPEED_OF_SOUND: return data.speedOfSound;
            case DataType::WIND_SPEED: return data.windSpeed;
        }
    }

    auto prev = std::prev(it);
    double ratio = (altitude - prev->altitude) / (it->altitude - prev->altitude);

    double val1 = 0.0, val2 = 0.0;
    switch(type) {
        case DataType::DENSITY:
            val1 = prev->density;
            val2 = it->density;
            break;
        case DataType::PRESSURE:
            val1 = prev->pressure;
            val2 = it->pressure;
            break;
        case DataType::TEMPERATURE:
            val1 = prev->temperature;
            val2 = it->temperature;
            break;
        case DataType::SPEED_OF_SOUND:
            val1 = prev->speedOfSound;
            val2 = it->speedOfSound;
            break;
        case DataType::WIND_SPEED:
            val1 = prev->windSpeed;
            val2 = it->windSpeed;
            break;
    }

    return val1 + ratio * (val2 - val1);
}

// Then modify the getter methods:
double Environment::getDensity(double altitude) const {
    return interpolateData(altitude, DataType::DENSITY);
}

double Environment::getPressure(double altitude) const {
    return interpolateData(altitude, DataType::PRESSURE);
}

double Environment::getTemperature(double altitude) const {
    return interpolateData(altitude, DataType::TEMPERATURE);
}

double Environment::getSpeedOfSound(double altitude) const {
    return interpolateData(altitude, DataType::SPEED_OF_SOUND);
}

double Environment::getWindSpeed(double altitude) const {
    return interpolateData(altitude, DataType::WIND_SPEED);
}

std::vector<double> Environment::getAltitudes() const {
    std::vector<double> altitudes;
    altitudes.reserve(atmosphericTable.size());
    for (const auto& data : atmosphericTable) {
        altitudes.push_back(data.altitude);
    }
    return altitudes;
}

double Environment::windSpeedChange(double gustSpeed, double gustLength, double distance) const {
    if (distance < 0) return 0.0;
    if (distance >= gustLength) return gustSpeed;
    return (gustSpeed/2.0) * (1.0 - std::cos((M_PI * distance) / gustLength));
}

double Environment::windSpeedGust(double gustSpeed, double gustLength, double distance) const {
    if (distance < 0) return 0.0;
    if (distance >= 3.0 * gustLength) return 0.0;
    
    if (distance < gustLength) {
        return (gustSpeed/2.0) * (1.0 - std::cos((M_PI * distance) / gustLength));
    } else if (distance < 2.0 * gustLength) {
        return gustSpeed;
    } else {
        return (gustSpeed/2.0) * (1.0 - std::cos((M_PI * distance/gustLength) + M_PI));
    }
}

void Environment::addWindParameter(double gustSpeedMean,
                                 double gustLengthMean, double gustLength3Sigma) {
    WindParameters param;
    param.gustSpeed = gustSpeedMean;
    param.gustLength = gustLengthMean;
    std::fill_n(param.direction, 3, 0.0);
    windParameters.push_back(param);
}

void Environment::removeWindParameter(size_t index) {
    if (index >= windParameters.size()) {
        throw std::out_of_range("Wind parameter index out of range");
    }
    windParameters.erase(windParameters.begin() + index);
    if (currentWindIndex >= windParameters.size()) {
        currentWindIndex = windParameters.empty() ? 0 : windParameters.size() - 1;
    }
}

Environment::WindParameters Environment::getWindParameter(size_t index) const {
    if (index >= windParameters.size()) {
        throw std::out_of_range("Wind parameter index out of range");
    }
    return windParameters[index];
}

void Environment::setWindParameterIndex(size_t index) {
    if (index >= windParameters.size()) {
        throw std::out_of_range("Wind parameter index out of range");
    }
    currentWindIndex = index;
}

std::pair<double, double> Environment::getWindAspects() const {
    if (windParameters.empty()) {
        return {0.0, 0.0};
    }
    const auto& param = windParameters[currentWindIndex];
    return {param.gustSpeed, param.gustLength};
}