#include "cuda_utils.cuh"
#include <cuda_fp16.h>

__global__ void swiglu_kernel(
    const half* input,
    half* output,
    int64_t size) {
    
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float x = __half2float(input[idx]);
    float gate = 1.0f / (1.0f + expf(-x));
    output[idx] = __float2half(x * gate);
}

void launch_swiglu(
    const half* input,
    half* output,
    int64_t size,
    cudaStream_t stream) {
    
    dim3 block(256);
    dim3 grid((size + block.x - 1) / block.x);
    swiglu_kernel<<<grid, block, 0, stream>>>(input, output, size);
}
