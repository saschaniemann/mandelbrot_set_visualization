cmake_minimum_required(VERSION 3.10)
project(mandelbrot_set_visualization LANGUAGES CXX CUDA)

find_package(CUDA 10.2 REQUIRED)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3)

# Find SDL2
find_package(SDL2 REQUIRED)
include_directories(${SDL2_INCLUDE_DIRS})

# include_directories(include)

# file(GLOB SOURCES "src/*.cu")

# Source files
set(SOURCES
    src/main.cpp
    src/kernel.cu
)

set_source_files_properties(src/main.cpp src/kernel.cu PROPERTIES LANGUAGE CUDA)

add_executable(mandelbrot_set_visualization ${SOURCES})

# Link libraries
target_link_libraries(mandelbrot_set_visualization ${SDL2_LIBRARIES} ${CUDA_LIBRARIES})
# target_include_directories(mandelbrot_set_visualization PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
