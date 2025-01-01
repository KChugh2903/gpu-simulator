# CMake generated Testfile for 
# Source directory: /home/chughster/Code/RocketSimulator
# Build directory: /home/chughster/Code/RocketSimulator
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(mpi_test "mpirun" "-np" "4" "/home/chughster/Code/RocketSimulator/rocket-sim")
set_tests_properties(mpi_test PROPERTIES  _BACKTRACE_TRIPLES "/home/chughster/Code/RocketSimulator/CMakeLists.txt;115;add_test;/home/chughster/Code/RocketSimulator/CMakeLists.txt;0;")
