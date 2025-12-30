gi#include "filter.cuh"
#include <vector>
namespace gpusql {
namespace cuda {

__global__ void filter_gt_kernel_atomic(
    const int32_t* __restrict__ input,
    int32_t* __restrict__ output,
    int32_t* __restrict__ count,
    int n,
    int32_t threshold,
    int max_output
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= n) return;
    
    int32_t value = input[idx];
    
    if (value > threshold) {
        int output_pos = atomicAdd(count, 1);
        
        if (output_pos < max_output) {
            output[output_pos] = idx; 
        }
    }
}

__global__ void filter_gt_kernel_warp(
    const int32_t* __restrict__ input,
    int32_t* __restrict__ output,
    int32_t* __restrict__ count,
    int n,
    int32_t threshold,
    int max_output
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;
    bool passes = false;
    if (idx < n) {
        passes = (input[idx] > threshold);
    }
    
    unsigned int ballot = __ballot_sync(0xFFFFFFFF, passes);
    
    int warp_count = __popc(ballot);
    
    int warp_offset = 0;
    if (lane == 0 && warp_count > 0) {
        warp_offset = atomicAdd(count, warp_count);
    }
    
    warp_offset = __shfl_sync(0xFFFFFFFF, warp_offset, 0);
    
    unsigned int mask_before = (1u << lane) - 1;
    int local_offset = __popc(ballot & mask_before);
    
    if (passes && idx < n) {
        int output_pos = warp_offset + local_offset;
        if (output_pos < max_output) {
            output[output_pos] = idx;
        }
    }
}

__global__ void filter_compute_flags_kernel(
    const int32_t* __restrict__ input,
    int32_t* __restrict__ flags,
    int n,
    int32_t threshold
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        flags[idx] = (input[idx] > threshold) ? 1 : 0;
    }
}

__global__ void filter_scatter_kernel(
    const int32_t* __restrict__ flags,
    const int32_t* __restrict__ positions,
    int32_t* __restrict__ output,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n && flags[idx] == 1) {
        output[positions[idx]] = idx;
    }
}

__global__ void filter_float_gt_kernel_atomic(
    const float* __restrict__ input,
    int32_t* __restrict__ output,
    int32_t* __restrict__ count,
    int n,
    float threshold,
    int max_output
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= n) return;
    
    float value = input[idx];
    
    if (value > threshold) {
        int output_pos = atomicAdd(count, 1);
        if (output_pos < max_output) {
            output[output_pos] = idx;
        }
    }
}

GpuFilterResult filter_greater_than_gpu(
    const int32_t* h_input,
    size_t n,
    int32_t threshold,
    bool use_warp_optimization
) {
    GpuFilterResult result = {};
    result.max_output = n; 

    int32_t* d_input = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input, n * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&result.d_output_indices, n * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&result.d_count, sizeof(int32_t)));
    
    CUDA_CHECK(cudaMemset(result.d_count, 0, sizeof(int32_t)));
    
    GpuTimer total_timer;
    total_timer.start();
    
    CUDA_CHECK(cudaMemcpy(d_input, h_input, n * sizeof(int32_t), cudaMemcpyHostToDevice));
    
    int block_size = DEFAULT_BLOCK_SIZE;
    int grid_size = grid_size_1d(n, block_size);
    
    GpuTimer kernel_timer;
    kernel_timer.start();
    
    if (use_warp_optimization) {
        filter_gt_kernel_warp<<<grid_size, block_size>>>(
            d_input, result.d_output_indices, result.d_count,
            static_cast<int>(n), threshold, static_cast<int>(n)
        );
    } else {
        filter_gt_kernel_atomic<<<grid_size, block_size>>>(
            d_input, result.d_output_indices, result.d_count,
            static_cast<int>(n), threshold, static_cast<int>(n)
        );
    }
    
    CUDA_CHECK_KERNEL();
    
    kernel_timer.stop();
    result.kernel_time_ms = kernel_timer.elapsed_ms();
    
    int32_t host_count = 0;
    CUDA_CHECK(cudaMemcpy(&host_count, result.d_count, sizeof(int32_t), cudaMemcpyDeviceToHost));
    result.count = static_cast<size_t>(host_count);
    
    total_timer.stop();
    result.total_time_ms = total_timer.elapsed_ms();
    
    CUDA_CHECK(cudaFree(d_input));
    
    return result;
}

GpuFilterResult filter_float_greater_gpu(
    const float* h_input,
    size_t n,
    float threshold
) {
    GpuFilterResult result = {};
    result.max_output = n;
    
    float* d_input = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&result.d_output_indices, n * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&result.d_count, sizeof(int32_t)));
    
    CUDA_CHECK(cudaMemset(result.d_count, 0, sizeof(int32_t)));
    
    GpuTimer total_timer;
    total_timer.start();
    
    CUDA_CHECK(cudaMemcpy(d_input, h_input, n * sizeof(float), cudaMemcpyHostToDevice));
    
    int block_size = DEFAULT_BLOCK_SIZE;
    int grid_size = grid_size_1d(n, block_size);
    
    GpuTimer kernel_timer;
    kernel_timer.start();
    
    filter_float_gt_kernel_atomic<<<grid_size, block_size>>>(
        d_input, result.d_output_indices, result.d_count,
        static_cast<int>(n), threshold, static_cast<int>(n)
    );
    
    CUDA_CHECK_KERNEL();
    
    kernel_timer.stop();
    result.kernel_time_ms = kernel_timer.elapsed_ms();
    
    int32_t host_count = 0;
    CUDA_CHECK(cudaMemcpy(&host_count, result.d_count, sizeof(int32_t), cudaMemcpyDeviceToHost));
    result.count = static_cast<size_t>(host_count);
    
    total_timer.stop();
    result.total_time_ms = total_timer.elapsed_ms();
    
    CUDA_CHECK(cudaFree(d_input));
    
    return result;
}

std::vector<int32_t> copy_filter_results_to_host(const GpuFilterResult& result) {
    std::vector<int32_t> indices(result.count);
    if (result.count > 0) {
        CUDA_CHECK(cudaMemcpy(
            indices.data(),
            result.d_output_indices,
            result.count * sizeof(int32_t),
            cudaMemcpyDeviceToHost
        ));
    }
    return indices;
}

void free_filter_result(GpuFilterResult& result) {
    if (result.d_output_indices) {
        cudaFree(result.d_output_indices);
        result.d_output_indices = nullptr;
    }
    if (result.d_count) {
        cudaFree(result.d_count);
        result.d_count = nullptr;
    }
}
} 
} 


