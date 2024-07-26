cmake_minimum_required(VERSION 3.10)
project(mandelbrot_set_visualization LANGUAGES CXX CUDA)

find_package(CUDA 10.2 REQUIRED)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3)

# # Find SDL2
# find_package(SDL2 REQUIRED)
# include_directories(${SDL2_INCLUDE_DIRS})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

find_package(Qt5 COMPONENTS Core Gui Widgets REQUIRED)


# include_directories(include)

# file(GLOB SOURCES "src/*.cu")

# Source files
set(SOURCES
    src/main.cpp
    src/kernel.cu
    src/TextureWidget.cpp
    # src/kernel.cuh 
    # src/TextureWidget.h
)

set_source_files_properties(src/TextureWidget.cpp src/kernel.cu PROPERTIES LANGUAGE CUDA)

add_executable(mandelbrot_set_visualization ${SOURCES})
get_property(core_options TARGET Qt5::Core PROPERTY INTERFACE_COMPILE_OPTIONS)
string(REPLACE "-fPIC" "" new_core_options ${core_options})
set_property(TARGET Qt5::Core PROPERTY INTERFACE_COMPILE_OPTIONS ${new_core_options})
set_property(TARGET Qt5::Core PROPERTY INTERFACE_POSITION_INDEPENDENT_CODE "ON")

# Link libraries
target_link_libraries(mandelbrot_set_visualization Qt5::Widgets Qt5::Core Qt5::Gui ${CUDA_LIBRARIES})
# target_include_directories(mandelbrot_set_visualization PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
