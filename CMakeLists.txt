cmake_minimum_required(VERSION 3.18)
project(tensorrt_cpp_api)

# Use ccache to speed up rebuilds
include(cmake/ccache.cmake)

# Set C++ version and optimization level
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Ofast -DNDEBUG -Wno-deprecated-declarations")

# For finding FindTensorRT.cmake
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

# For clangd LSP
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Use the correct version of CUDA
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)

# We require CUDA, OpenCV, and TensorRT
find_package(TensorRT REQUIRED)
# Eign3 is required for OpenCV built with Eigen3
find_package(Eigen3 REQUIRED)
if(EIGEN3_FOUND)
    include_directories(${EIGEN3_INCLUDE_DIR})
else()
    message(FATAL_ERROR "Eigen3 was not found. Run 'sudo apt-get install libeigen3-dev'.")
endif()

# Only use required components from OpenCV
find_package(OpenCV REQUIRED COMPONENTS
    core imgproc imgcodecs cudaarithm cudaimgproc cudawarping
)

add_library(tensorrt_cpp_api SHARED
    src/engine.cpp
)

target_include_directories(tensorrt_cpp_api PUBLIC ${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${TensorRT_INCLUDE_DIRS} include include/interfaces)
target_link_libraries(tensorrt_cpp_api PUBLIC ${OpenCV_LIBS} ${CUDA_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${TensorRT_LIBRARIES})

add_executable(run_inference_benchmark
    src/benchmark.cpp
)
target_link_libraries(run_inference_benchmark tensorrt_cpp_api)

if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "11")
    # Use gcc <= 8 for CUDA 10.2 compiliation
    set(CMAKE_CUDA_HOST_COMPILER /usr/bin/g++-8)
endif()

enable_language(CUDA)