// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <cuda.h>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <thread>

struct Ticket {
    uint32_t* flag;
    unsigned timeout_ms;
    std::mutex mutex;
    std::condition_variable cv;
    bool cancel = false;
    std::atomic<int> state{0}; // 0 armed, 1 observed release, 2 timeout, 3 cancelled
    std::thread worker;
};

extern "C" int gate_map(uint64_t host, uint64_t* device) {
    CUdeviceptr pointer = 0;
    auto result = cuMemHostGetDevicePointer(&pointer, reinterpret_cast<void*>(host), 0);
    *device = pointer;
    return static_cast<int>(result);
}
extern "C" int gate_wait(uint64_t stream, uint64_t pointer) {
    return static_cast<int>(cuStreamWaitValue32(reinterpret_cast<CUstream>(stream), pointer, 1,
                                               CU_STREAM_WAIT_VALUE_EQ));
}
extern "C" int gate_write(uint64_t stream, uint64_t pointer) {
    return static_cast<int>(cuStreamWriteValue32(reinterpret_cast<CUstream>(stream), pointer, 1,
                                                CU_STREAM_WRITE_VALUE_DEFAULT));
}
extern "C" void gate_host_open(uint64_t pointer) {
    __atomic_store_n(reinterpret_cast<uint32_t*>(pointer), 1U, __ATOMIC_SEQ_CST);
}
extern "C" int gate_host_is_open(uint64_t pointer) {
    return __atomic_load_n(reinterpret_cast<uint32_t*>(pointer), __ATOMIC_SEQ_CST) == 1U;
}
extern "C" int gate_pool_internal(int value, int* previous) {
    CUdevice device;
    CUresult result = cuCtxGetDevice(&device);
    if (result != CUDA_SUCCESS) return static_cast<int>(result);
    CUmemoryPool pool;
    result = cuDeviceGetDefaultMemPool(&pool, device);
    if (result != CUDA_SUCCESS) return static_cast<int>(result);
    result = cuMemPoolGetAttribute(pool, CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES, previous);
    if (result != CUDA_SUCCESS) return static_cast<int>(result);
    return static_cast<int>(cuMemPoolSetAttribute(pool, CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES, &value));
}
extern "C" void* gate_watchdog(uint64_t pointer, unsigned timeout_ms) {
    Ticket* ticket = nullptr;
    try {
        ticket = new Ticket;
        ticket->flag = reinterpret_cast<uint32_t*>(pointer);
        ticket->timeout_ms = timeout_ms;
        ticket->worker = std::thread([ticket] {
            auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(ticket->timeout_ms);
            std::unique_lock<std::mutex> lock(ticket->mutex);
            for (;;) {
                if (ticket->cancel) { ticket->state.store(3); return; }
                if (__atomic_load_n(ticket->flag, __ATOMIC_SEQ_CST) == 1U) {
                    ticket->state.store(1); return;
                }
                if (std::chrono::steady_clock::now() >= deadline) {
                    // CPU release needs neither the GIL nor a runnable CUDA queue.
                    ticket->state.store(2);
                    __atomic_store_n(ticket->flag, 1U, __ATOMIC_SEQ_CST);
                    return;
                }
                ticket->cv.wait_for(lock, std::chrono::milliseconds(1));
            }
        });
        return ticket;
    } catch (...) {
        delete ticket;
        return nullptr;
    }
}
extern "C" int gate_watchdog_state(void* raw) {
    return reinterpret_cast<Ticket*>(raw)->state.load();
}
extern "C" void gate_watchdog_close(void* raw) {
    auto ticket = reinterpret_cast<Ticket*>(raw);
    {
        std::lock_guard<std::mutex> lock(ticket->mutex);
        ticket->cancel = true;
    }
    ticket->cv.notify_all();
    if (ticket->worker.joinable()) ticket->worker.join();
    delete ticket;
}
