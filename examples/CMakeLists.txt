# 示例程序构建
add_executable(moe_example
    moe_example.cpp
)

target_include_directories(moe_example PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(moe_example PRIVATE
    moe_core
    moe_cuda_kernels
    moe_cpu_kernels
)

if(ENABLE_MULTI_GPU)
    target_link_libraries(moe_example PRIVATE NCCL::nccl)
endif()
