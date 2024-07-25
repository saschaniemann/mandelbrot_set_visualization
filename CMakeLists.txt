cmake_minimum_required(VERSION 3.10)
project(mandelbrot_set_visualization)

find_package(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3)

# Find SDL2
find_package(SDL2 REQUIRED)
include_directories(${SDL2_INCLUDE_DIRS})

include_directories(include)

# file(GLOB SOURCES "src/*.cu")

# Source files
set(SOURCES
    src/main.cpp
    src/kernel.cu
)

cuda_add_executable(mandelbrot_set_visualization ${SOURCES})

# Link libraries
target_link_libraries(cuda_project ${SDL2_LIBRARIES} CUDA::cudart)
