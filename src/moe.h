#pragma once
#include <vector>
#include <memory>
#include <cstdint>

namespace moe {
namespace cpu {
    void sparse_matmul(
        const float* input,
        const float* weights,
        const int* expert_indices,
        float* output,
        int batch_size,
        int input_dim,
        int output_dim,
        int experts_count);

    void rms_norm(
        const float* input,
        const float* weight,
        float* output,
        float epsilon,
        int batch_size,
        int hidden_size);
}

// 设备类型枚举
enum class DeviceType {
    CPU,
    CUDA
};

// 张量描述符
struct TensorDesc {
    std::vector<int64_t> shape;
    DeviceType device;
    int device_id = 0;
};

// 专家路由接口
class IExpertRouter {
public:
    virtual ~IExpertRouter() = default;
    virtual std::vector<int> route(const float* input, int batch_size) = 0;
};

// 多卡通信接口
class IMultiGPUComm {
public:
    virtual ~IMultiGPUComm() = default;
    virtual void all_reduce(float* data, size_t size) = 0;
    virtual void broadcast(float* data, size_t size, int root) = 0;
};

// MOE层接口
class MOELayer {
public:
    virtual TensorDesc forward(const TensorDesc& input) = 0;
    virtual ~MOELayer() = default;
};

// 框架初始化
void initialize(int num_gpus = 1);
void shutdown();

} // namespace moe
