cmake_minimum_required(VERSION 3.18)
project(MOE_Engine LANGUAGES C CXX CUDA)

# 基础编译选项
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES "native")

# 查找CUDA工具包
find_package(CUDAToolkit REQUIRED)

# 包含路径
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# 第三方依赖
find_package(Threads REQUIRED)

# 多卡支持配置
option(ENABLE_MULTI_GPU "Enable multi-GPU support" ON)
if(ENABLE_MULTI_GPU)
    find_package(NCCL QUIET)
    if(NCCL_FOUND)
        message(STATUS "Found NCCL: ${NCCL_LIBRARIES}")
        add_definitions(-DUSE_NCCL)
    else()
        message(WARNING "NCCL not found, disabling multi-GPU support")
        set(ENABLE_MULTI_GPU OFF)
    endif()
endif()

# 子目录
add_subdirectory(src)
add_subdirectory(kernels)
add_subdirectory(examples)
