// CPU emulation shim: lets cuda/intref.cu run on the host so the kernel logic (indexing, reductions,
// shared memory use) can be tested without a GPU. Each block's threads run as real threads when the
// kernel uses __syncthreads, otherwise serially.
#pragma once
#include <barrier>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <thread>
#include <vector>
#include <chrono>

#define __global__
#define __device__
#define __host__
#define __forceinline__ inline
#define __shared__ static

struct dim3 { unsigned x, y, z; dim3(unsigned a = 1, unsigned b = 1, unsigned c = 1) : x(a), y(b), z(c) {} };
thread_local dim3 threadIdx, blockIdx, blockDim, gridDim;
static std::barrier<> *g_barrier = nullptr;
static void __syncthreads() { g_barrier->arrive_and_wait(); }
static int64_t *g_dyn_shared = nullptr;

typedef int cudaError_t;
enum { cudaSuccess = 0 };
enum cudaMemcpyKind { cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost };
struct cudaDeviceProp { char name[64]; int major, minor; size_t totalGlobalMem; };
typedef int cudaEvent_t;
static const char *cudaGetErrorString(cudaError_t) { return "emulated error"; }
static cudaError_t cudaGetLastError() { return 0; }
static cudaError_t cudaMalloc(void **p, size_t n) { *p = malloc(n); return *p ? 0 : 1; }
template <class T> static cudaError_t cudaMalloc(T **p, size_t n) { return cudaMalloc((void **)p, n); }
static cudaError_t cudaMemcpy(void *d, const void *s, size_t n, cudaMemcpyKind) { memcpy(d, s, n); return 0; }
static cudaError_t cudaGetDeviceCount(int *n) { *n = 1; return 0; }
static cudaError_t cudaGetDeviceProperties(cudaDeviceProp *p, int) { strcpy(p->name, "CPU emulation"); p->major = 0; p->minor = 0; p->totalGlobalMem = 1ull << 33; return 0; }
static int g_ev = 0;
static void cudaEventCreate(cudaEvent_t *e) { *e = g_ev++; }
static std::chrono::steady_clock::time_point g_t0;
static void cudaEventRecord(cudaEvent_t e) { if (e == 0) g_t0 = std::chrono::steady_clock::now(); }
static void cudaEventSynchronize(cudaEvent_t) {}
static void cudaEventElapsedTime(float *ms, cudaEvent_t, cudaEvent_t) { *ms = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - g_t0).count(); }

template <class... P, class... A>
static void emu_launch(void (*k)(P...), bool uses_sync, dim3 g, dim3 b, size_t shm_bytes, A... a) {
    std::vector<int64_t> dyn(shm_bytes / 8 + 1);
    g_dyn_shared = dyn.data();
    for (unsigned by = 0; by < g.y; by++)
        for (unsigned bx = 0; bx < g.x; bx++) {
            auto set = [&](unsigned tx) { blockIdx = dim3(bx, by, 0); threadIdx = dim3(tx, 0, 0); blockDim = b; gridDim = g; };
            if (!uses_sync) {
                for (unsigned tx = 0; tx < b.x; tx++) { set(tx); k(a...); }
            } else {
                std::barrier<> bar(b.x);
                g_barrier = &bar;
                std::vector<std::thread> th;
                for (unsigned tx = 0; tx < b.x; tx++) th.emplace_back([&, tx] { set(tx); k(a...); });
                for (auto &t : th) t.join();
            }
        }
}
