#include "utils.cuh"

namespace gpusql {
namespace cuda {
void initialize_cuda() {
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
}
} 
} 


