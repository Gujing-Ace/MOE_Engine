#include "moe.h"
#include <vector>
#include <stdexcept>
#include <cuda_runtime.h>

#ifdef USE_NCCL
#include <nccl.h>
#endif

namespace moe {

namespace {
    std::vector<int> gpu_devices;
    bool initialized = false;
#ifdef USE_NCCL
    ncclComm_t nccl_comm;
#endif
}

void initialize(int num_gpus) {
    if (initialized) return;
    
    // 检测可用GPU设备
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    
    if (num_gpus > device_count) {
        throw std::runtime_error("Requested more GPUs than available");
    }
    
    gpu_devices.resize(num_gpus);
    for (int i = 0; i < num_gpus; ++i) {
        gpu_devices[i] = i % device_count;
    }
    
#ifdef USE_NCCL
    // 初始化NCCL通信
    ncclUniqueId id;
    if (num_gpus > 1) {
        ncclGetUniqueId(&id);
        ncclCommInitAll(&nccl_comm, num_gpus, gpu_devices.data());
    }
#endif

    initialized = true;
}

void shutdown() {
    if (!initialized) return;
    
#ifdef USE_NCCL
    if (gpu_devices.size() > 1) {
        ncclCommDestroy(nccl_comm);
    }
#endif

    initialized = false;
}

} // namespace moe
