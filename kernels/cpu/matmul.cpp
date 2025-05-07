#include "cpu_utils.h"
#include <cmath>
#include <numeric>

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
    int experts_count) {
    
    ThreadPool pool;
    
    auto compute_expert = [&](int bid) {
        int expert_id = expert_indices[bid];
        if (expert_id < 0 || expert_id >= experts_count) return;
        
        const float* expert_weights = weights + expert_id * input_dim * output_dim;
        float* expert_output = output + bid * output_dim;
        
        #ifdef USE_AVX2
        // AVX2优化实现
        for (int i = 0; i < output_dim; i += 8) {
            __m256 sum = _mm256_setzero_ps();
            for (int j = 0; j < input_dim; ++j) {
                __m256 w = _mm256_load_ps(expert_weights + j * output_dim + i);
                __m256 x = _mm256_set1_ps(input[bid * input_dim + j]);
                sum = _mm256_add_ps(sum, _mm256_mul_ps(x, w));
            }
            _mm256_store_ps(expert_output + i, sum);
        }
        #else
        // 标量实现
        for (int i = 0; i < output_dim; ++i) {
            float sum = 0.0f;
            for (int j = 0; j < input_dim; ++j) {
                sum += input[bid * input_dim + j] * 
                       expert_weights[j * output_dim + i];
            }
            expert_output[i] = sum;
        }
        #endif
    };
    
    // 并行处理batch
    parallel_for(0, batch_size, compute_expert);
}

} // namespace cpu
} // namespace moe
