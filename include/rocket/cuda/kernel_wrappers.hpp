// kernel_wrappers.hpp
#pragma once
#include <vector>
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include "helper_math.h"

// Error checking macro
#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in file '%s' in line %i : %s.\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

// Kernel launch wrappers
void launchIntegrateStatesKernel(
    const void* states, 
    void* nextStates, 
    int numStates, 
    float dt, 
    cudaStream_t stream = nullptr
);

void launchAerodynamicsKernel(
    const void* states, 
    void* forces, 
    void* moments, 
    int numStates, 
    cudaStream_t stream = nullptr
);

void launchEnvironmentKernel(
    const void* positions, 
    void* densities, 
    void* pressures, 
    void* winds, 
    int numPoints, 
    cudaStream_t stream = nullptr
);

class CUDAKernelWrapper {
public:
    // Structures matching the CUDA kernel data structures
    struct Vector3 {
        float x, y, z;
        Vector3() : x(0), y(0), z(0) {}
        Vector3(float _x, float _y, float _z) : x(_x), y(_y), z(_z) {}
    };

    struct Quaternion {
        float w, x, y, z;
        Quaternion() : w(1), x(0), y(0), z(0) {}
        Quaternion(float _w, float _x, float _y, float _z) : w(_w), x(_x), y(_y), z(_z) {}
    };

    struct RocketState {
        Vector3 position;
        Vector3 velocity;
        Quaternion orientation;
        Vector3 angularVelocity;
    };
    CUDAKernelWrapper();
    ~CUDAKernelWrapper();
    CUDAKernelWrapper(const CUDAKernelWrapper&) = delete;
    CUDAKernelWrapper& operator=(const CUDAKernelWrapper&) = delete;
    void initialize(size_t maxBatchSize = 1024);
    void computeDynamicsBatch(
        const std::vector<RocketState>& states,
        std::vector<RocketState>& nextStates,
        float dt
    );

    void computeAerodynamicsBatch(
        const std::vector<RocketState>& states,
        std::vector<Vector3>& forces,
        std::vector<Vector3>& moments
    );

    void computeEnvironmentBatch(
        const std::vector<Vector3>& positions,
        std::vector<float>& densities,
        std::vector<float>& pressures,
        std::vector<Vector3>& winds
    );

private:
    bool initialized;
    size_t maxBatchSize;
    cudaStream_t computeStream;
    void* d_states;
    void* d_nextStates;
    void* d_forces;
    void* d_moments;
    void* d_densities;
    void* d_pressures;
    void* d_winds;
    void cleanup();
    void checkInitialization() const;
    void validateBatchSize(size_t size) const;
};


