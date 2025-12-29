cmake_minimum_required(VERSION 3.18)

project(gpu_sql LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "89")
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(CUDAToolkit REQUIRED)

# Core Library (CPU code)
add_library(gpu_sql_core STATIC
    src/core/column.cpp
    src/core/table.cpp
    src/cpu/baseline.cpp
)

target_include_directories(gpu_sql_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

if(MSVC)
    target_compile_options(gpu_sql_core PRIVATE /W3 /O2)
endif()

# CUDA Kernels Library
add_library(gpu_sql_kernels STATIC
    src/kernels/utils.cu
    src/kernels/filter.cu
)

target_include_directories(gpu_sql_kernels PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(gpu_sql_kernels PUBLIC
    CUDA::cudart
)

set_target_properties(gpu_sql_kernels PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
)

# Main Executable
add_executable(gpu_sql_benchmark
    src/main.cpp
)

target_link_libraries(gpu_sql_benchmark PRIVATE
    gpu_sql_core
    gpu_sql_kernels
)

if(MSVC)
    target_compile_options(gpu_sql_benchmark PRIVATE /W3 /O2)
endif()

message(STATUS "========================================")
message(STATUS "GPU-SQL Build Configuration")
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "========================================")
