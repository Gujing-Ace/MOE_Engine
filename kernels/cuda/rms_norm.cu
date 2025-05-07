#include "cuda_utils.cuh"
#include <cuda_fp16.h>
#include <cmath>

namespace moe {
namespace cuda {

template<typename T>
__global__ void rms_norm_kernel(
    const T* input,
    const T* weight,
    T* output,
    float epsilon,
    int batch_size,
    int hidden_size) {
    
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    
    if (bid >= batch_size) return;
    
    __shared__ float variance;
    const T* x = input + bid * hidden_size;
    T* y = output + bid * hidden_size;
    
    // 计算平方和
    float sum = 0.0f;
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float val = static_cast<float>(x[i]);
        sum += val * val;
    }
    
    // 规约求和
    sum = blockReduceSum(sum);
    if (tid == 0) {
        variance = rsqrtf(sum / hidden_size + epsilon);
    }
    __syncthreads();
    
    // 归一化并应用权重
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        y[i] = static_cast<T>(static_cast<float>(x[i]) * variance * static_cast<float>(weight[i]));
    }
}

// 规约求和辅助函数
__device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__device__ float blockReduceSum(float val) {
    __shared__ float shared[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    
    val = warpReduceSum(val);
    
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    
    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0.0f;
    val = warpReduceSum(val);
    
    return val;
}

void rms_norm(
    const float* input,
    const float* weight,
    float* output,
    float epsilon,
    int batch_size,
    int hidden_size,
    cudaStream_t stream = 0) {
    
    dim3 blocks(batch_size);
    dim3 threads(256);
    
    rms_norm_kernel<float><<<blocks, threads, 0, stream>>>(
        input, weight, output, epsilon, batch_size, hidden_size);
    
    CUDA_CHECK(cudaGetLastError());
}

} // namespace cuda
} // namespace moe
