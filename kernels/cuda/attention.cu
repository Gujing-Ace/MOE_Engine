#include "cuda_utils.cuh"
#include <cuda_fp16.h>

__global__ void attention_kernel(
    const half* query,
    const half* key,
    const half* value,
    half* output,
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

    // 计算QK^T
    float score = 0.0f;
    for (int j = 0; j < seq_len; ++j) {
        for (int k = 0; k < head_dim; ++k) {
            float q = __half2float(query[((b * num_heads + h) * seq_len + i) * head_dim + k]);
            float k_val = __half2float(key[((b * num_heads + h) * seq_len + j) * head_dim + k]);
            score += q * k_val;
        }
    }
    score /= sqrtf(head_dim);

    // Softmax
    float sum_exp = 0.0f;
    for (int j = 0; j < seq_len; ++j) {
        sum_exp += expf(score);
    }
    float attention_weight = expf(score) / sum_exp;

    // 计算加权和
    float out_val = 0.0f;
    for (int j = 0; j < seq_len; ++j) {
        for (int k = 0; k < head_dim; ++k) {
            float v = __half2float(value[((b * num_heads + h) * seq_len + j) * head_dim + k]);
            out_val += attention_weight * v;
        }
    }

    output[idx] = __float2half(out_val);
}

void launch_attention(
    const half* query,
    const half* key,
    const half* value,
    half* output,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    cudaStream_t stream) {

    int total_elements = batch_size * num_heads * seq_len * head_dim;
    dim3 block(256);
    dim3 grid((total_elements + block.x - 1) / block.x);
    attention_kernel<<<grid, block, 0, stream>>>(
        query, key, value, output,
        batch_size, num_heads, seq_len, head_dim);
}
