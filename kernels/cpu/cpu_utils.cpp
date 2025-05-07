#include "cpu_utils.h"
#include <stdexcept>

namespace moe {
namespace cpu {

ThreadPool::ThreadPool(size_t threads) {
    if(threads == 0) {
        throw std::invalid_argument("Thread count must be positive");
    }
    
    for(size_t i = 0; i < threads; ++i) {
        workers.emplace_back([this] {
            while(true) {
                std::function<void()> task;
                {
                    std::unique_lock<std::mutex> lock(queue_mutex);
                    condition.wait(lock, [this] { 
                        return stop || !tasks.empty(); 
                    });
                    
                    if(stop && tasks.empty()) {
                        return;
                    }
                    
                    task = std::move(tasks.back());
                    tasks.pop_back();
                }
                task();
            }
        });
    }
}

ThreadPool::~ThreadPool() {
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        stop = true;
    }
    condition.notify_all();
    for(auto& worker : workers) {
        worker.join();
    }
}

void* aligned_alloc(size_t size, size_t alignment) {
#ifdef _WIN32
    return _aligned_malloc(size, alignment);
#else
    return ::aligned_alloc(alignment, size);
#endif
}

void aligned_free(void* ptr) {
#ifdef _WIN32
    _aligned_free(ptr);
#else
    ::free(ptr);
#endif
}

} // namespace cpu
} // namespace moe
