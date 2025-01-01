// kernel_wrappers.cpp
#include "kernel_wrappers.hpp"
#include "cuda_runtime.h"

// Forward declarations of actual CUDA kernels (to be implemented in .cu files)
extern "C" {
    void integrateStatesKernel_wrapper(
        const void* states, 
        void* nextStates, 
        int numStates, 
        float dt
    );

    void computeAerodynamicsKernel_wrapper(
        const void* states, 
        void* forces, 
        void* moments, 
        int numStates
    );

    void computeEnvironmentKernel_wrapper(
        const void* positions, 
        void* densities, 
        void* pressures, 
        void* winds, 
        int numPoints
    );
}

void launchIntegrateStatesKernel(
    const void* states, 
    void* nextStates, 
    int numStates, 
    float dt, 
    cudaStream_t stream
) {
    // Synchronize the stream if needed
    if (stream) {
        cudaStreamSynchronize(stream);
    }

    // Call the kernel wrapper
    integrateStatesKernel_wrapper(
        states, 
        nextStates, 
        numStates, 
        dt
    );

    // Check for kernel launch errors
    cudaError_t cudaError = cudaGetLastError();
    if (cudaError != cudaSuccess) {
        throw std::runtime_error(
            "Failed to launch integrateStatesKernel: " + 
            std::string(cudaGetErrorString(cudaError))
        );
    }
}

void launchAerodynamicsKernel(
    const void* states, 
    void* forces, 
    void* moments, 
    int numStates, 
    cudaStream_t stream
) {
    // Synchronize the stream if needed
    if (stream) {
        cudaStreamSynchronize(stream);
    }

    // Call the kernel wrapper
    computeAerodynamicsKernel_wrapper(
        states, 
        forces, 
        moments, 
        numStates
    );

    // Check for kernel launch errors
    cudaError_t cudaError = cudaGetLastError();
    if (cudaError != cudaSuccess) {
        throw std::runtime_error(
            "Failed to launch computeAerodynamicsKernel: " + 
            std::string(cudaGetErrorString(cudaError))
        );
    }
}

void launchEnvironmentKernel(
    const void* positions, 
    void* densities, 
    void* pressures, 
    void* winds, 
    int numPoints, 
    cudaStream_t stream
) {
    // Synchronize the stream if needed
    if (stream) {
        cudaStreamSynchronize(stream);
    }

    // Call the kernel wrapper
    computeEnvironmentKernel_wrapper(
        positions, 
        densities, 
        pressures, 
        winds, 
        numPoints
    );

    // Check for kernel launch errors
    cudaError_t cudaError = cudaGetLastError();
    if (cudaError != cudaSuccess) {
        throw std::runtime_error(
            "Failed to launch computeEnvironmentKernel: " + 
            std::string(cudaGetErrorString(cudaError))
        );
    }
}
