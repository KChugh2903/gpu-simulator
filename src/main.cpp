
// main.cpp with MPI example
#include "mpi_monte_carlo.hpp"
#include <iostream>

int main(int argc, char** argv) {
    MPIMonteCarlo::initializeMPI(argc, argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    try {
        MonteCarlo::Parameters params;
        MPIMonteCarlo mpiMC(params);
        int totalRuns = 1000;
        auto results = mpiMC.runDistributedSimulations(totalRuns);
        if (rank == 0) {
            std::cout << "Completed " << results.size() << " simulations\n";
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error on rank " << rank << ": " << e.what() << std::endl;
    }
    
    // Finalize MPI
    MPIMonteCarlo::finalizeMPI();
    return 0;
}