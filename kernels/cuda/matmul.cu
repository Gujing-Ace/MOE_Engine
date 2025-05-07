#include "cuda_utils.cuh"
#include <cuda_fp16.h>

namespace moe {
namespace cuda {

__global__ void sparse_matmul_kernel(
    const float* input,
    const float* weights,
    const int* expert_indices,
    float* output,
    int batch_size,
    int input_dim,
    int output_dim,
    int experts_count) {
    
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    
    if (bid >= batch_size) return;
    
    int expert_id = expert_indices[bid];
    if (expert_id < 0 || expert_id >= experts_count) return;
    
    const float* expert_weights = weights + expert_id * input_dim * output_dim;
    float* expert_output = output + bid * output_dim;
    
    for (int i = tid; i < output_dim; i += blockDim.x) {
        float sum = 0.0f;
        for (int j = 0; j < input_dim; ++j) {
            sum += input[bid * input_dim + j] * 
                   expert_weights[j * output_dim + i];
        }
        expert_output[i] = sum;
    }
}

void sparse_matmul(
    const float* input,
    const float* weights,
    const int* expert_indices,
    float* output,
    int batch_size,
    int input_dim,
    int output_dim,
    int experts_count,
    cudaStream_t stream = 0) {
    
    dim3 blocks(batch_size);
    dim3 threads(256);
    
    sparse_matmul_kernel<<<blocks, threads, 0, stream>>>(
        input, weights, expert_indices, output,
        batch_size, input_dim, output_dim, experts_count);
    
    CUDA_CHECK(cudaGetLastError());
}

} // namespace cuda
} // namespace moe
