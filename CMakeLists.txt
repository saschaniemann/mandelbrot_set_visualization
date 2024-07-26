cmake_minimum_required(VERSION 3.10)
project(mandelbrot_set_visualization LANGUAGES CXX CUDA)

find_package(CUDA 10.2 REQUIRED)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

find_package(Qt5 COMPONENTS Core Gui Widgets REQUIRED)

# Source files
set(SOURCES
    src/main.cpp
    src/kernel.cu
    src/TextureWidget.cpp
)

set_source_files_properties(src/TextureWidget.cpp src/kernel.cu PROPERTIES LANGUAGE CUDA)

add_executable(mandelbrot_set_visualization ${SOURCES})
get_property(core_options TARGET Qt5::Core PROPERTY INTERFACE_COMPILE_OPTIONS)
string(REPLACE "-fPIC" "" new_core_options ${core_options})
set_property(TARGET Qt5::Core PROPERTY INTERFACE_COMPILE_OPTIONS ${new_core_options})
set_property(TARGET Qt5::Core PROPERTY INTERFACE_POSITION_INDEPENDENT_CODE "ON")

# Link libraries
target_link_libraries(mandelbrot_set_visualization Qt5::Widgets Qt5::Core Qt5::Gui ${CUDA_LIBRARIES})
