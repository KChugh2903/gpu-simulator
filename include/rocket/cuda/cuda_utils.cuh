// cuda_utils.cuh
#pragma once
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_math.h>

__constant__ float d_gravity = 9.81f;
__constant__ float d_dt;
__constant__ float d_currentTime;

__constant__ float d_altitudeTable[1000];
__constant__ float d_densityTable[1000];
__constant__ float d_windTable[1000];

__constant__ float d_airDensity;
__constant__ float d_machNumber;

// Replace our check function with a different macro name
#define CUDA_CHECK_ERROR(val) \
    do { \
        cudaError_t err = val; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\".\n", \
                    __FILE__, __LINE__, static_cast<unsigned int>(err), \
                    cudaGetErrorString(err), #val); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Common CUDA structures
struct CUDAVector3 {
    float x, y, z;
    
    __host__ __device__ CUDAVector3() : x(0), y(0), z(0) {}
    __host__ __device__ CUDAVector3(float _x, float _y, float _z) : x(_x), y(_y), z(_z) {}
    
    __host__ __device__ float norm() const {
        return sqrtf(x*x + y*y + z*z);
    }
};

struct CUDAQuaternion {
    float w, x, y, z;
    
    __host__ __device__ CUDAQuaternion() : w(1), x(0), y(0), z(0) {}
    __host__ __device__ CUDAQuaternion(float _w, float _x, float _y, float _z) 
        : w(_w), x(_x), y(_y), z(_z) {}
};

// CUDA state structure
struct CUDARocketState {
    CUDAVector3 position;
    CUDAVector3 velocity;
    CUDAQuaternion orientation;
    CUDAVector3 angularVelocity;
    float alpha;
    float beta;
};

__global__ void integrateStatesKernel(
    const float* states, 
    float* nextStates, 
    int numStates, 
    float dt
);

__global__ void aerodynamicsKernel(
    const float* states,
    float* forces,
    float* moments,
    int numStates
);

__global__ void environmentKernel(
    const float* positions,
    float* densities,
    float* pressures,
    float* winds,
    int numPoints
);