// mpi_monte_carlo.hpp
#pragma once
#include <mpi.h>
#include "monte_carlo.hpp"
#include <vector>

class MPIMonteCarlo {
public:
    MPIMonteCarlo(const MonteCarlo::Parameters& params);
    ~MPIMonteCarlo();
    std::vector<MonteCarlo::SimulationResult> runDistributedSimulations(int totalRuns);
    void gatherResults();

    // MPI utility functions
    static void initializeMPI(int argc, char** argv);
    static void finalizeMPI();

private:
    int rank;        
    int numProcesses;    
    MonteCarlo mc;  
    MPI_Datatype mpiStateType;
    MPI_Datatype mpiResultType;
    void createMPITypes();
    void distributeWork(int totalRuns, int& localStart, int& localCount);
    std::vector<MonteCarlo::SimulationResult> gatherSimulationResults(
        const std::vector<MonteCarlo::SimulationResult>& localResults);
};