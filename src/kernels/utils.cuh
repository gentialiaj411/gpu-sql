#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace gpusql {
namespace cuda {

#define CUDA_CHECK(call)                                                        \
    do {                                                                         \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA Error at %s:%d: %s\n",                        \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            throw std::runtime_error(std::string("CUDA Error: ") +              \
                                     cudaGetErrorString(err));                  \
        }                                                                        \
    } while (0)

#define CUDA_CHECK_KERNEL()                                                     \
    do {                                                                         \
        CUDA_CHECK(cudaGetLastError());                                         \
    } while (0)

#define CUDA_SYNC_CHECK()                                                       \
    do {                                                                         \
        CUDA_CHECK(cudaDeviceSynchronize());                                    \
    } while (0)

inline void print_device_info() {
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    
    if (device_count == 0) {
        printf("No CUDA-capable devices found!\n");
        return;
    }
    
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║                    GPU DEVICE INFO                           ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n");
    
    for (int i = 0; i < device_count; ++i) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
        
        printf("\nDevice %d: %s\n", i, prop.name);
        printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("  Multiprocessors:    %d\n", prop.multiProcessorCount);
        printf("  Global Memory:      %.2f GB\n", 
               prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        printf("  Shared Memory/Block: %zu KB\n", prop.sharedMemPerBlock / 1024);
        printf("  Max Threads/Block:  %d\n", prop.maxThreadsPerBlock);
        printf("  Warp Size:          %d\n", prop.warpSize);
    }
    printf("\n");
}

inline std::pair<size_t, size_t> get_memory_info() {
    size_t free_mem, total_mem;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    return {free_mem, total_mem};
}

inline void print_memory_usage() {
    auto [free_mem, total_mem] = get_memory_info();
    size_t used_mem = total_mem - free_mem;
    printf("GPU Memory: %.2f / %.2f GB (%.1f%% used)\n",
           used_mem / (1024.0 * 1024.0 * 1024.0),
           total_mem / (1024.0 * 1024.0 * 1024.0),
           100.0 * used_mem / total_mem);
}

constexpr int DEFAULT_BLOCK_SIZE = 256;
constexpr int SMALL_BLOCK_SIZE = 128;
constexpr int LARGE_BLOCK_SIZE = 512;

inline int grid_size_1d(size_t n, int block_size = DEFAULT_BLOCK_SIZE) {
    return static_cast<int>((n + block_size - 1) / block_size);
}

inline dim3 make_grid(size_t n, int block_size = DEFAULT_BLOCK_SIZE) {
    return dim3(grid_size_1d(n, block_size));
}

inline dim3 make_block(int block_size = DEFAULT_BLOCK_SIZE) {
    return dim3(block_size);
}

template<typename T>
class DeviceBuffer {
public:
    
    explicit DeviceBuffer(size_t n) : size_(n), ptr_(nullptr) {
        if (n > 0) {
            CUDA_CHECK(cudaMalloc(&ptr_, n * sizeof(T)));
        }
    }
    
    DeviceBuffer(const T* host_data, size_t n) : DeviceBuffer(n) {
        copy_from_host(host_data);
    }
    
    DeviceBuffer(DeviceBuffer&& other) noexcept 
        : size_(other.size_), ptr_(other.ptr_) {
        other.ptr_ = nullptr;
        other.size_ = 0;
    }
    
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            free();
            size_ = other.size_;
            ptr_ = other.ptr_;
            other.ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }
    
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    
    ~DeviceBuffer() {
        free();
    }
    
    T* get() { return ptr_; }
    const T* get() const { return ptr_; }
    size_t size() const { return size_; }
    size_t bytes() const { return size_ * sizeof(T); }
    bool empty() const { return size_ == 0; }
    
    void copy_from_host(const T* host_data) {
        if (size_ > 0 && host_data != nullptr) {
            CUDA_CHECK(cudaMemcpy(ptr_, host_data, bytes(), cudaMemcpyHostToDevice));
        }
    }
    
    void copy_to_host(T* host_data) const {
        if (size_ > 0 && host_data != nullptr) {
            CUDA_CHECK(cudaMemcpy(host_data, ptr_, bytes(), cudaMemcpyDeviceToHost));
        }
    }

    void zero() {
        if (size_ > 0) {
            CUDA_CHECK(cudaMemset(ptr_, 0, bytes()));
        }
    }

private:
    size_t size_;
    T* ptr_;
    
    void free() {
        if (ptr_ != nullptr) {
            cudaFree(ptr_);  
            ptr_ = nullptr;
        }
        size_ = 0;
    }
};

class GpuTimer {
public:
    GpuTimer() {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }
    
    ~GpuTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    
    GpuTimer(const GpuTimer&) = delete;
    GpuTimer& operator=(const GpuTimer&) = delete;
    
    void start(cudaStream_t stream = 0) {
        CUDA_CHECK(cudaEventRecord(start_, stream));
    }
    
    void stop(cudaStream_t stream = 0) {
        CUDA_CHECK(cudaEventRecord(stop_, stream));
    }
    
    float elapsed_ms() {
        CUDA_CHECK(cudaEventSynchronize(stop_));
        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }

private:
    cudaEvent_t start_, stop_;
};
} 
} 


