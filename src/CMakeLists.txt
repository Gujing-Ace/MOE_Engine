# 核心框架构建配置
add_library(moe_core STATIC
    init.cpp
)

target_include_directories(moe_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(moe_core PRIVATE 
    moe_cuda_kernels
    moe_cpu_kernels
    Threads::Threads
)

if(ENABLE_MULTI_GPU)
    target_link_libraries(moe_core PRIVATE NCCL::nccl)
endif()
