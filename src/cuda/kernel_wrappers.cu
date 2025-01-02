// kernel_wrappers.cu
#include "cuda_utils.cuh"
#include <cuda_runtime.h>

__device__ void quaternionMultiply(const CUDAQuaternion& q1, const CUDAQuaternion& q2, CUDAQuaternion& result) {
    result.w = q1.w*q2.w - q1.x*q2.x - q1.y*q2.y - q1.z*q2.z;
    result.x = q1.w*q2.x + q1.x*q2.w + q1.y*q2.z - q1.z*q2.y;
    result.y = q1.w*q2.y - q1.x*q2.z + q1.y*q2.w + q1.z*q2.x;
    result.z = q1.w*q2.z + q1.x*q2.y - q1.y*q2.x + q1.z*q2.w;
}

__global__ void integrateStatesKernel(
    const CUDARocketState* states,
    CUDARocketState* nextStates,
    int numStates
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numStates) return;
    float dt = 0.01;
    const CUDARocketState& state = states[idx];
    CUDARocketState& next = nextStates[idx];

    next.position.x = state.position.x + state.velocity.x * dt;
    next.position.y = state.position.y + state.velocity.y * dt;
    next.position.z = state.position.z + state.velocity.z * dt;

    next.velocity.x = state.velocity.x;
    next.velocity.y = state.velocity.y;
    next.velocity.z = state.velocity.z - 9.8 * dt;

    float halfDt = 0.5f * dt;
    CUDAQuaternion omega(0,
        state.angularVelocity.x * halfDt,
        state.angularVelocity.y * halfDt,
        state.angularVelocity.z * halfDt
    );
    CUDAQuaternion temp;
    quaternionMultiply(omega, state.orientation, temp);
    next.orientation = temp;
    next.angularVelocity = state.angularVelocity;
}


__global__ void computeAerodynamicsKernel(
    const CUDARocketState* states,
    CUDAVector3* forces,
    CUDAVector3* moments,
    int numStates
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numStates) return;

    const CUDARocketState& state = states[idx];
    float velocity = state.velocity.norm();
    float dynamicPressure = 0.5f * d_airDensity * velocity * velocity;
    
    forces[idx].x = -dynamicPressure * cosf(state.alpha);
    forces[idx].y = -dynamicPressure * sinf(state.alpha);
    forces[idx].z = -dynamicPressure * sinf(state.beta);

    moments[idx].x = 0.0f;  // Roll moment
    moments[idx].y = -dynamicPressure * sinf(state.alpha); 
    moments[idx].z = -dynamicPressure * sinf(state.beta);   
}

__global__ void computeEnvironmentKernel(
    const CUDAVector3* positions,
    float* densities,
    float* pressures,
    CUDAVector3* winds,
    int numPoints
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numPoints) return;
    float altitude = positions[idx].z;
    int tableIdx = 0;
    while (tableIdx < 999 && d_altitudeTable[tableIdx + 1] < altitude) tableIdx++;
    
    float t = (altitude - d_altitudeTable[tableIdx]) / 
              (d_altitudeTable[tableIdx + 1] - d_altitudeTable[tableIdx]);
    densities[idx] = d_densityTable[tableIdx] + 
                    t * (d_densityTable[tableIdx + 1] - d_densityTable[tableIdx]);
    
    pressures[idx] = 101325.0f * expf(-altitude / 7400.0f);
    float windSpeed = d_windTable[tableIdx] + 
                     t * (d_windTable[tableIdx + 1] - d_windTable[tableIdx]);
    winds[idx] = CUDAVector3(windSpeed, 0.0f, 0.0f);
}

extern "C" {
    void integrateStatesKernel_wrapper(
        const void* states, 
        void* nextStates, 
        int numStates, 
        float dt
    ) {
        // Calculate grid and block dimensions
        int threadsPerBlock = 256;
        int blocksPerGrid = (numStates + threadsPerBlock - 1) / threadsPerBlock;

        // Copy dt to constant memory
        cudaMemcpyToSymbol(d_dt, &dt, sizeof(float));

        // Launch kernel
        integrateStatesKernel<<<blocksPerGrid, threadsPerBlock>>>(
            static_cast<const CUDARocketState*>(states),
            static_cast<CUDARocketState*>(nextStates), 
            numStates
        );
    }

    void computeAerodynamicsKernel_wrapper(
        const void* states, 
        void* forces, 
        void* moments, 
        int numStates
    ) {
        // Calculate grid and block dimensions
        int threadsPerBlock = 256;
        int blocksPerGrid = (numStates + threadsPerBlock - 1) / threadsPerBlock;

        // Launch kernel
        computeAerodynamicsKernel<<<blocksPerGrid, threadsPerBlock>>>(
            static_cast<const CUDARocketState*>(states),
            static_cast<CUDAVector3*>(forces),
            static_cast<CUDAVector3*>(moments),
            numStates
        );
    }

    void computeEnvironmentKernel_wrapper(
        const void* positions, 
        void* densities, 
        void* pressures, 
        void* winds, 
        int numPoints
    ) {
        int threadsPerBlock = 256;
        int blocksPerGrid = (numPoints + threadsPerBlock - 1) / threadsPerBlock;
        computeEnvironmentKernel<<<blocksPerGrid, threadsPerBlock>>>(
            static_cast<const CUDAVector3*>(positions),
            static_cast<float*>(densities),
            static_cast<float*>(pressures),
            static_cast<CUDAVector3*>(winds),
            numPoints
        );
    }
}