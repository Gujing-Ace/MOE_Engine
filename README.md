# MOE Engine

混合专家(MoE)模型的高性能推理框架，项目仍在构建中。

## 功能特性
- 支持专家并行(Expert Parallelism) 
- CUDA加速核心算子
- 多GPU NCCL通信支持
- 模块化设计，易于扩展

## 构建指南

### 依赖项
- CUDA 11.0+
- CMake 3.18+
- NCCL (可选，多GPU支持)

### 构建步骤
```bash
mkdir build && cd build
cmake .. -DENABLE_MULTI_GPU=ON  # 启用多GPU支持
make -j
```

## 使用示例

```cpp
#include "moe.h"

int main() {
    moe::initialize(2); // 初始化2个GPU
    // 使用MOE模型...
    moe::shutdown();
    return 0;
}
```

## 目录结构
```
.
├── CMakeLists.txt      # 顶层构建配置
├── src/               # 核心框架代码
├── kernels/           # CPU/CUDA算子实现
├── examples/          # 示例代码
└── build/             # 构建目录
```

## 许可证
Apache License 2.0
