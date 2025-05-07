#include "cuda_utils.cuh"
#include <cuda_fp16.h>
#include <algorithm>

__global__ void topk_gating_kernel(
    const half* logits,
    int* expert_indices,
    half* expert_weights,
    int batch_size,
    int num_experts,
    int k) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    // 使用共享内存存储每个样本的logits
    extern __shared__ half shared_logits[];
    for (int i = threadIdx.x; i < num_experts; i += blockDim.x) {
        shared_logits[i] = logits[idx * num_experts + i];
    }
    __syncthreads();

    // 每个线程处理一个样本的Top-K选择
    for (int i = 0; i < k; ++i) {
        half max_val = __float2half(-INFINITY);
        int max_idx = 0;

        // 找出当前最大值
        for (int j = 0; j < num_experts; ++j) {
            if (shared_logits[j] > max_val) {
                max_val = shared_logits[j];
                max_idx = j;
            }
        }

        // 存储结果并屏蔽已选专家
        expert_indices[idx * k + i] = max_idx;
        expert_weights[idx * k + i] = max_val;
        shared_logits[max_idx] = __float2half(-INFINITY);
    }

    // Softmax归一化权重
    float sum_exp = 0.0f;
    for (int i = 0; i < k; ++i) {
        sum_exp += expf(__half2float(expert_weights[idx * k + i]));
    }
    for (int i = 0; i < k; ++i) {
        expert_weights[idx * k + i] = __float2half(
            expf(__half2float(expert_weights[idx * k + i])) / sum_exp);
    }
}

void launch_topk_gating(
    const half* logits,
    int* expert_indices,
    half* expert_weights,
    int batch_size,
    int num_experts,
    int k,
    cudaStream_t stream) {

    dim3 block(256);
    dim3 grid((batch_size + block.x - 1) / block.x);
    size_t shared_mem_size = num_experts * sizeof(half);
    
    topk_gating_kernel<<<grid, block, shared_mem_size, stream>>>(
        logits, expert_indices, expert_weights,
        batch_size, num_experts, k);
}
