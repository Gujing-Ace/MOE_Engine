#include "moe.h"
#include <iostream>
#include <vector>
#include <random>

int main() {
    // 初始化框架
    moe::initialize(1); // 使用1个GPU
    
    // 模拟输入数据 (batch=4, dim=256)
    const int batch_size = 4;
    const int input_dim = 256;
    const int output_dim = 512;
    const int experts_count = 8;
    
    std::vector<float> input(batch_size * input_dim);
    std::vector<float> weights(experts_count * input_dim * output_dim);
    std::vector<int> expert_indices(batch_size);
    
    // 随机初始化
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::uniform_int_distribution<int> expert_dist(0, experts_count-1);
    
    for (auto& v : input) v = dist(gen);
    for (auto& v : weights) v = dist(gen);
    for (auto& v : expert_indices) v = expert_dist(gen);
    
    // 分配输出内存
    std::vector<float> output(batch_size * output_dim);
    
    // 执行稀疏矩阵乘法 (CPU版本)
    moe::cpu::sparse_matmul(
        input.data(),
        weights.data(),
        expert_indices.data(),
        output.data(),
        batch_size,
        input_dim,
        output_dim,
        experts_count
    );
    
    // 准备RMSNorm参数
    std::vector<float> norm_weights(output_dim, 1.0f);
    std::vector<float> norm_output(batch_size * output_dim);
    
    // 执行RMSNorm (CPU版本)
    moe::cpu::rms_norm(
        output.data(),
        norm_weights.data(),
        norm_output.data(),
        1e-5f,
        batch_size,
        output_dim
    );
    
    std::cout << "MOE推理完成，输出维度: " 
              << batch_size << "x" << output_dim << std::endl;
    
    // 清理资源
    moe::shutdown();
    return 0;
}
