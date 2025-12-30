#pragma once
#include "utils.cuh"
#include <cstdint>
#include <vector>
namespace gpusql {
namespace cuda {

struct GpuFilterResult {
    int32_t* d_output_indices; 
    int32_t* d_count;        
    size_t max_output;     
    size_t count;              
    float kernel_time_ms;      
    float total_time_ms;  
};

__global__ void filter_gt_kernel_atomic(
    const int32_t* __restrict__ input,
    int32_t* __restrict__ output,
    int32_t* __restrict__ count,
    int n,
    int32_t threshold,
    int max_output
);

__global__ void filter_gt_kernel_warp(
    const int32_t* __restrict__ input,
    int32_t* __restrict__ output,
    int32_t* __restrict__ count,
    int n,
    int32_t threshold,
    int max_output
);

__global__ void filter_compute_flags_kernel(
    const int32_t* __restrict__ input,
    int32_t* __restrict__ flags,
    int n,
    int32_t threshold
);

__global__ void filter_scatter_kernel(
    const int32_t* __restrict__ flags,
    const int32_t* __restrict__ positions, 
    int32_t* __restrict__ output,
    int n
);

__global__ void filter_float_gt_kernel_atomic(
    const float* __restrict__ input,
    int32_t* __restrict__ output,
    int32_t* __restrict__ count,
    int n,
    float threshold,
    int max_output
);

GpuFilterResult filter_greater_than_gpu(
    const int32_t* h_input,
    size_t n,
    int32_t threshold,
    bool use_warp_optimization = true
);

GpuFilterResult filter_float_greater_gpu(
    const float* h_input,
    size_t n,
    float threshold
);

std::vector<int32_t> copy_filter_results_to_host(const GpuFilterResult& result);

void free_filter_result(GpuFilterResult& result);

} 
} 


