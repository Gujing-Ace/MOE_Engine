#pragma once
#include <cuda_runtime.h>
#include <stdexcept>

#define CUDA_CHECK(cmd) do {                         \
    cudaError_t e = cmd;                             \
    if(e != cudaSuccess) {                           \
        printf("CUDA error %s:%d '%s'\n",            \
            __FILE__,__LINE__,cudaGetErrorString(e)); \
        exit(EXIT_FAILURE);                          \
    }                                               \
} while(0)

namespace moe {
namespace cuda {

// 设备内存分配器
template<typename T>
T* device_alloc(size_t count) {
    T* ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&ptr, count * sizeof(T)));
    return ptr;
}

// 设备内存释放
template<typename T>
void device_free(T* ptr) {
    CUDA_CHECK(cudaFree(ptr));
}

// 内存拷贝
template<typename T>
void copy_to_device(T* dst, const T* src, size_t count) {
    CUDA_CHECK(cudaMemcpy(dst, src, count * sizeof(T), cudaMemcpyHostToDevice));
}

template<typename T>
void copy_to_host(T* dst, const T* src, size_t count) {
    CUDA_CHECK(cudaMemcpy(dst, src, count * sizeof(T), cudaMemcpyDeviceToHost));
}

// 设备同步
inline void sync_device() {
    CUDA_CHECK(cudaDeviceSynchronize());
}

template<typename T>
__device__ T warpReduceSum(T val) {
    for (int offset = warpSize/2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// 块内归约求和
template<typename T>
__device__ T blockReduceSum(T val) {
    static __shared__ T shared[32];
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    val = warpReduceSum(val);

    if (lane == 0) shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;
    if (wid == 0) val = warpReduceSum(val);
    return val;
}

} // namespace cuda
} // namespace moe
