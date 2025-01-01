cmake_minimum_required(VERSION 3.18)
project(rocket-sim LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find required packages
find_package(CUDA REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenMP REQUIRED)

# Configure CUDA
set(CMAKE_CUDA_ARCHITECTURES 75)
enable_language(CUDA)

# Include directories
include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include/rocket/core
    ${PROJECT_SOURCE_DIR}/include/rocket/cuda
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/include/third_party/cuda-samples/Common
    ${EIGEN3_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIRS}
    /opt/cuda/targets/x86_64-linux/include
)

# Source files
file(GLOB_RECURSE CORE_SOURCES 
    "src/core/*.cpp"
)

file(GLOB_RECURSE CUDA_SOURCES 
    "src/cuda/*.cu"
    "src/cuda/*.cpp"
)

# Set CUDA specific compile options
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fopenmp")

# Create the executable
add_executable(rocket-sim
    ${CORE_SOURCES}
    ${CUDA_SOURCES}
)

# Link libraries
target_link_libraries(rocket-sim
    PRIVATE
    Eigen3::Eigen
    OpenMP::OpenMP_CXX
    ${CUDA_LIBRARIES}
    cudart
)

# Set compile options
target_compile_options(rocket-sim PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic -O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)

# Set output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)