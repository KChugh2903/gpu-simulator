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
find_package(MPI REQUIRED)   # Add MPI package
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(glfw3 REQUIRED)

# Configure CUDA
set(CMAKE_CUDA_ARCHITECTURES 75)
enable_language(CUDA)

# Include directories
include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include/rocket/core
    ${PROJECT_SOURCE_DIR}/include/rocket/cuda
    ${PROJECT_SOURCE_DIR}/include/rocket/mpi
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/include/third_party/cuda-samples/Common
    ${EIGEN3_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIRS}
    ${MPI_CXX_INCLUDE_DIRS}    # Add MPI include directories
    /opt/cuda/targets/x86_64-linux/include
    ${PROJECT_SOURCE_DIR}/include/rocket/visualization
    ${OPENGL_INCLUDE_DIRS}
    ${GLEW_INCLUDE_DIRS}
)

# Replace find_package(GLM REQUIRED) with:
include_directories(/usr/include/glm)

file(GLOB_RECURSE VIS_SOURCES 
    "src/visualization/*.cpp"
)

# Source files
file(GLOB_RECURSE CORE_SOURCES 
    "src/core/*.cpp"
    "src/*.cpp"
)

file(GLOB_RECURSE CUDA_SOURCES 
    "src/cuda/*.cu"
    "src/cuda/*.cpp"
)

file(GLOB_RECURSE MPI_SOURCES 
    "src/mpi/*.cpp"
)

# Set CUDA specific compile options
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fopenmp")

# Create the executable
add_executable(rocket-sim
    ${CORE_SOURCES}
    ${CUDA_SOURCES}
    ${MPI_SOURCES} 
    ${VIS_SOURCES}
)

# Link libraries
target_link_libraries(rocket-sim
    PRIVATE
    Eigen3::Eigen
    OpenMP::OpenMP_CXX
    ${CUDA_LIBRARIES}
    cudart
    MPI::MPI_CXX
    OpenGL::GL
    GLEW::GLEW
    glfw
)

# Set compile options
target_compile_options(rocket-sim PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic -O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)

# Add MPI compile definitions
target_compile_definitions(rocket-sim PRIVATE
    OMPI_SKIP_MPICXX        # Skip deprecated C++ bindings
    ${MPI_CXX_COMPILE_DEFINITIONS}
)

# Set MPI compile options
target_compile_options(rocket-sim PRIVATE
    ${MPI_CXX_COMPILE_OPTIONS}
)

# Set output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Optional: Add MPI wrapper compiler
if(MPI_CXX_COMPILER)
    set(CMAKE_CXX_COMPILER ${MPI_CXX_COMPILER})
endif()

# Optional: Add testing support
enable_testing()
add_test(NAME mpi_test
         COMMAND mpirun -np 4 $<TARGET_FILE:rocket-sim>)

add_custom_command(TARGET rocket-sim POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_SOURCE_DIR}/src/visualization/shaders
    $<TARGET_FILE_DIR:rocket-sim>/shaders
)