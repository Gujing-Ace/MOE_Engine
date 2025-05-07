#include "cpu_utils.h"
#include <cmath>
#include <numeric>

namespace moe {
namespace cpu {

void rms_norm(
    const float* input,
    const float* weight,
    float* output,
    float epsilon,
    int batch_size,
    int hidden_size) {
    
    ThreadPool pool;
    
    auto compute_norm = [&](int bid) {
        const float* x = input + bid * hidden_size;
        float* y = output + bid * hidden_size;
        
        // 计算平方和
        float sum = 0.0f;
        #ifdef USE_AVX2
        __m256 sum_vec = _mm256_setzero_ps();
        for (int i = 0; i < hidden_size; i += 8) {
            __m256 v = _mm256_load_ps(x + i);
            sum_vec = _mm256_add_ps(sum_vec, _mm256_mul_ps(v, v));
        }
        sum = _mm256_reduce_add_ps(sum_vec);
        #else
        for (int i = 0; i < hidden_size; ++i) {
            sum += x[i] * x[i];
        }
        #endif
        
        // 计算归一化因子
        float variance = 1.0f / sqrtf(sum / hidden_size + epsilon);
        
        // 应用归一化和权重
        #ifdef USE_AVX2
        __m256 var_vec = _mm256_set1_ps(variance);
        for (int i = 0; i < hidden_size; i += 8) {
            __m256 v = _mm256_load_ps(x + i);
            __m256 w = _mm256_load_ps(weight + i);
            _mm256_store_ps(y + i, _mm256_mul_ps(_mm256_mul_ps(v, var_vec), w));
        }
        #else
        for (int i = 0; i < hidden_size; ++i) {
            y[i] = x[i] * variance * weight[i];
        }
        #endif
    };
    
    // 并行处理batch
    parallel_for(0, batch_size, compute_norm);
}

} // namespace cpu
} // namespace moe
