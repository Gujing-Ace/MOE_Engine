# CUDA kernels构建配置
add_library(moe_cuda_kernels STATIC
    cuda_utils.cuh
    matmul.cu
    rms_norm.cu
    swiglu.cu
    attention.cu
    rope.cu
    gating.cu
)

target_include_directories(moe_cuda_kernels PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(moe_cuda_kernels PRIVATE CUDA::cudart)

if(ENABLE_MULTI_GPU)
    target_link_libraries(moe_cuda_kernels PRIVATE NCCL::nccl)
    target_compile_definitions(moe_cuda_kernels PRIVATE USE_NCCL)
endif()
