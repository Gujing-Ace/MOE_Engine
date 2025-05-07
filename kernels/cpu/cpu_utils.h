#pragma once
#include <vector>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>

namespace moe {
namespace cpu {

// 线程池实现
class ThreadPool {
public:
    explicit ThreadPool(size_t threads = std::thread::hardware_concurrency());
    ~ThreadPool();
    
    template<class F>
    void enqueue(F&& f) {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            tasks.emplace_back(std::forward<F>(f));
        }
        condition.notify_one();
    }

private:
    std::vector<std::thread> workers;
    std::vector<std::function<void()>> tasks;
    
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop = false;
};

// SIMD优化宏
#ifdef __AVX2__
#include <immintrin.h>
#define USE_AVX2
#endif

// 内存对齐分配
void* aligned_alloc(size_t size, size_t alignment);
void aligned_free(void* ptr);

// 并行for循环
template<typename Func>
void parallel_for(int begin, int end, Func f, int grain = 1024) {
    int range = end - begin;
    if (range <= grain) {
        for (int i = begin; i < end; ++i) f(i);
        return;
    }
    
    int mid = begin + range / 2;
    std::thread t([=] { parallel_for(mid, end, f, grain); });
    parallel_for(begin, mid, f, grain);
    t.join();
}

} // namespace cpu
} // namespace moe
