#include "mpi_monte_carlo.hpp"
#include <iostream>

MPIMonteCarlo::MPIMonteCarlo(const MonteCarlo::Parameters& params)
    : mc(params) {
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);
    createMPITypes();
}

MPIMonteCarlo::~MPIMonteCarlo() {
    MPI_Type_free(&mpiStateType);
    MPI_Type_free(&mpiResultType);
}

void MPIMonteCarlo::initializeMPI(int argc, char** argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) {
        std::cerr << "Warning: MPI implementation does not support MPI_THREAD_FUNNELED\n";
    }
}

void MPIMonteCarlo::finalizeMPI() {
    MPI_Finalize();
}

void MPIMonteCarlo::createMPITypes() {
    {
        MPI_Datatype types[] = {MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE};
        int blocklengths[] = {3, 3, 4, 3};  
        MPI_Aint offsets[4];
        
        Dynamics::State temp;
        MPI_Get_address(&temp.position, &offsets[0]);
        MPI_Get_address(&temp.velocity, &offsets[1]);
        MPI_Get_address(&temp.quaternion, &offsets[2]);
        MPI_Get_address(&temp.angularVel, &offsets[3]);
        
        MPI_Aint base;
        MPI_Get_address(&temp, &base);
        for(int i = 0; i < 4; i++) {
            offsets[i] = MPI_Aint_diff(offsets[i], base);
        }
        
        MPI_Type_create_struct(4, blocklengths, offsets, types, &mpiStateType);
        MPI_Type_commit(&mpiStateType);
    }
    
    {
        MPI_Datatype types[] = {mpiStateType, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE};
        int blocklengths[] = {1, 1, 1, 1}; 
        MPI_Aint offsets[4];
        
        MonteCarlo::SimulationResult temp;
        MPI_Get_address(&temp.trajectory, &offsets[0]);
        MPI_Get_address(&temp.maxAltitude, &offsets[1]);
        MPI_Get_address(&temp.maxVelocity, &offsets[2]);
        MPI_Get_address(&temp.flightTime, &offsets[3]);
        
        MPI_Aint base;
        MPI_Get_address(&temp, &base);
        for(int i = 0; i < 4; i++) {
            offsets[i] = MPI_Aint_diff(offsets[i], base);
        }
        
        MPI_Type_create_struct(4, blocklengths, offsets, types, &mpiResultType);
        MPI_Type_commit(&mpiResultType);
    }
}

void MPIMonteCarlo::distributeWork(int totalRuns, int& localStart, int& localCount) {
    int runsPerProcess = totalRuns / numProcesses;
    int remainingRuns = totalRuns % numProcesses;
    if (rank < remainingRuns) {
        localCount = runsPerProcess + 1;
        localStart = rank * localCount;
    } else {
        localCount = runsPerProcess;
        localStart = (remainingRuns * (runsPerProcess + 1)) + 
                    ((rank - remainingRuns) * runsPerProcess);
    }
}

std::vector<MonteCarlo::SimulationResult> MPIMonteCarlo::runDistributedSimulations(int totalRuns) {
    int localStart, localCount;
    distributeWork(totalRuns, localStart, localCount);
    std::vector<MonteCarlo::SimulationResult> localResults = mc.runSimulations(localCount);
    return gatherSimulationResults(localResults);
}

std::vector<MonteCarlo::SimulationResult> MPIMonteCarlo::gatherSimulationResults(
    const std::vector<MonteCarlo::SimulationResult>& localResults) {
    std::vector<MonteCarlo::SimulationResult> allResults;
    int localSize = localResults.size();
    std::vector<int> allSizes(numProcesses);
    
    MPI_Gather(&localSize, 1, MPI_INT,
               allSizes.data(), 1, MPI_INT,
               0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        int totalSize = 0;
        for (int size : allSizes) {
            totalSize += size;
        }
        allResults.resize(totalSize);
        std::vector<int> displacements(numProcesses);
        int currentDisp = 0;
        for (int i = 0; i < numProcesses; i++) {
            displacements[i] = currentDisp;
            currentDisp += allSizes[i];
        }
        MPI_Gatherv(localResults.data(), localSize, mpiResultType,
                    allResults.data(), allSizes.data(), displacements.data(),
                    mpiResultType, 0, MPI_COMM_WORLD);
    } else {
        MPI_Gatherv(localResults.data(), localSize, mpiResultType,
                    nullptr, nullptr, nullptr,
                    mpiResultType, 0, MPI_COMM_WORLD);
    }
    
    return allResults;
}
