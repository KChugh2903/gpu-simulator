// mpi_monte_carlo.hpp
#pragma once
#include <mpi.h>
#include "monte_carlo.hpp"
#include <vector>

class MPIMonteCarlo {
public:
    MPIMonteCarlo(const MonteCarlo::Parameters& params);
    ~MPIMonteCarlo();

    // Run distributed simulations
    std::vector<MonteCarlo::SimulationResult> runDistributedSimulations(int totalRuns);

    // Collect and aggregate results
    void gatherResults();

    // MPI utility functions
    static void initializeMPI(int argc, char** argv);
    static void finalizeMPI();

private:
    int rank;               // Current process rank
    int numProcesses;       // Total number of processes
    MonteCarlo mc;         // Local Monte Carlo instance
    
    // MPI data types for our structures
    MPI_Datatype mpiStateType;
    MPI_Datatype mpiResultType;
    
    // Helper methods
    void createMPITypes();
    void distributeWork(int totalRuns, int& localStart, int& localCount);
    std::vector<MonteCarlo::SimulationResult> gatherSimulationResults(
        const std::vector<MonteCarlo::SimulationResult>& localResults);
};