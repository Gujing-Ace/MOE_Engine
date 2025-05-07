#include "cuda_utils.cuh"
#include <cuda_fp16.h>
#include <math.h>

__device__ void apply_rope(
    half* x,
    half* out,
    int pos,
    int dim,
    int head_dim) {

    float theta = 10000.0f;
    float freq = pos / powf(theta, 2.0f * (dim % (head_dim/2)) / head_dim);
    
    float cos_val = cosf(freq);
    float sin_val = sinf(freq);
    
    float x0 = __half2float(x[2*dim]);
    float x1 = __half2float(x[2*dim+1]);
    
    out[2*dim] = __float2half(x0 * cos_val - x1 * sin_val);
    out[2*dim+1] = __float2half(x0 * sin_val + x1 * cos_val);
}

__global__ void rope_kernel(
    const half* input,
    half* output,
    const int* positions,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * num_heads * seq_len * head_dim) return;

    int b = idx / (num_heads * seq_len * head_dim);
    int h = (idx / (seq_len * head_dim)) % num_heads;
    int i = (idx / head_dim) % seq_len;
    int d = idx % head_dim;

    int pos = positions[b * seq_len + i];
    apply_rope(
        (half*)&input[idx],
        (half*)&output[idx],
        pos,
        d,
        head_dim);
}

void launch_rope(
    const half* input,
    half* output,
    const int* positions,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    cudaStream_t stream) {

    int total_elements = batch_size * num_heads * seq_len * head_dim;
    dim3 block(256);
    dim3 grid((total_elements + block.x - 1) / block.x);
    rope_kernel<<<grid, block, 0, stream>>>(
        input, output, positions,
        batch_size, num_heads, seq_len, head_dim);
}
