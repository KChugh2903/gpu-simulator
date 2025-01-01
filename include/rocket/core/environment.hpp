// environment.hpp
#pragma once
#include <Eigen/Dense>
#include <vector>
#include <memory>

class Environment {
public:
    Environment(double latitude, double longitude, double elevation, bool useDefaultData = true);
    ~Environment() = default;
    struct AtmosphericData {
        double altitude;
        double density;
        double pressure;
        double temperature;
        double speedOfSound;
        double windSpeed;
    };
    struct WindParameters {
        double gustSpeed;  
        double gustLength;   
        double direction[3];  
    };
    double getDensity(double altitude) const;
    double getPressure(double altitude) const;
    double getTemperature(double altitude) const;
    double getSpeedOfSound(double altitude) const;
    double getWindSpeed(double altitude) const;
    std::vector<double> getAltitudes() const;
    double windSpeedChange(double gustSpeed, double gustLength, double distance) const;
    double windSpeedGust(double gustSpeed, double gustLength, double distance) const;
    std::pair<double, double> getWindAspects() const;
    void addWindParameter(double gustSpeedMean,  
                         double gustLengthMean, double gustLength3Sigma);
    void removeWindParameter(size_t index);
    WindParameters getWindParameter(size_t index) const;
    void setWindParameterIndex(size_t index);

private:
    double latitude;
    double longitude;
    double elevation;
    std::vector<AtmosphericData> atmosphericTable;
    std::vector<WindParameters> windParameters;
    size_t currentWindIndex;
    void initializeAtmosphericData(bool useDefaultData);
    void loadDefaultData();
    void generateEnvironmentData();
    enum class DataType {
        DENSITY,
        PRESSURE,
        TEMPERATURE,
        SPEED_OF_SOUND,
        WIND_SPEED
    };
    double interpolateData(double altitude, DataType type) const;
};