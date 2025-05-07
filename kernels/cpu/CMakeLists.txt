# CPU kernels构建配置
add_library(moe_cpu_kernels STATIC
    cpu_utils.cpp
    cpu_utils.h
    matmul.cpp
    rms_norm.cpp
)

target_include_directories(moe_cpu_kernels PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(moe_cpu_kernels PRIVATE Threads::Threads)

if(ENABLE_MULTI_GPU)
    target_compile_definitions(moe_cpu_kernels PRIVATE USE_NCCL)
endif()
