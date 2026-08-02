// ═══════════════════════════════════════════════════════════════════════════
//  ncu 프로파일링용 독립 실행 하네스
//
//  torch 없이 커널만 돌린다. Makefile 이 커널마다 하나씩 컴파일한다:
//
//      nvcc -DMK_KERNEL_SRC='"kernel_3.cu"' -DMK_NS=kernel_3 \
//           profile/harness.cu -o bin/kernel_3
//
//  실행 흐름:
//      1) 데이터 할당 + 채우기
//      2) warmup 회 실행 (클럭/캐시 안정화)
//      3) warmup 회 더 실행하며 cudaEvent 로 평균 시간 측정 (참고 출력용)
//      4) cudaProfilerStart()  ->  단 1회 실행  ->  cudaProfilerStop()
//
//  ncu 는 --profile-from-start off 로 실행하므로 4)의 1회만 잡힌다.
//  --set full 은 커널을 여러 번 replay 하지만, 대상은 그 1회 launch 다.
// ═══════════════════════════════════════════════════════════════════════════

//단독으로 컴파일되지 않도록 설계되어 있음
#ifndef MK_KERNEL_SRC
#error "MK_KERNEL_SRC 가 필요합니다. 예: -DMK_KERNEL_SRC='\"kernel_1.cu\"'"
#endif
#ifndef MK_NS
#error "MK_NS 가 필요합니다. 예: -DMK_NS=kernel_1"
#endif

#define MK_STR2(x) #x
#define MK_STRINGIFY(x) MK_STR2(x)

// kernel_N.cu 의 torch 바인딩을 빼고 __global__ 커널만 가져온다.
#define MK_STANDALONE 1
#include MK_KERNEL_SRC

#include <cuda_fp16.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(expr)                                                       \
    do {                                                                       \
        cudaError_t _e = (expr);                                               \
        if (_e != cudaSuccess) {                                               \
            std::fprintf(stderr, "CUDA error: %s\n  at %s:%d\n",               \
                         cudaGetErrorString(_e), __FILE__, __LINE__);          \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

int main(int argc, char **argv) {
    using Cfg = MK_NS::Cfg;
    using value_t = half;

    const int seq_len = argc > 1 ? std::atoi(argv[1]) : 4096;
    const int batch = argc > 2 ? std::atoi(argv[2]) : 4;
    const int n_heads = argc > 3 ? std::atoi(argv[3]) : 16;
    const int warmup = argc > 4 ? std::atoi(argv[4]) : 10;

    constexpr int d_head = Cfg::d_head;

    if (seq_len % Cfg::B_r != 0 || seq_len % Cfg::B_c != 0) {
        std::fprintf(stderr, "seq_len(%d) 은 B_r(%d), B_c(%d) 의 배수여야 합니다\n",
                     seq_len, Cfg::B_r, Cfg::B_c);
        return 1;
    }

    // ── 데이터 준비 ─────────────────────────────────────────────────────────
    const size_t n_elems =
        static_cast<size_t>(batch) * seq_len * n_heads * d_head;
    const size_t n_bytes = n_elems * sizeof(value_t);

    void *dQ, *dK, *dV, *dO;
    CUDA_CHECK(cudaMalloc(&dQ, n_bytes));
    CUDA_CHECK(cudaMalloc(&dK, n_bytes));
    CUDA_CHECK(cudaMalloc(&dV, n_bytes));
    CUDA_CHECK(cudaMalloc(&dO, n_bytes));

    {
        std::vector<value_t> host(n_elems);
        uint32_t state = 12345u;
        auto fill = [&](void *dst) {
            for (size_t i = 0; i < n_elems; ++i) {
                state = state * 1664525u + 1013904223u;
                const float u = static_cast<float>(state >> 8) / 16777216.0f;
                host[i] = __float2half(u * 2.0f - 1.0f); // [-1, 1)
            }
            CUDA_CHECK(cudaMemcpy(dst, host.data(), n_bytes,
                                  cudaMemcpyHostToDevice));
        };
        fill(dQ);
        fill(dK);
        fill(dV);
    }

    // ── 런치 구성 ───────────────────────────────────────────────────────────
    // 텐서 레이아웃 (batch, seq, heads, d_head) contiguous
    const mk::KernelArgs args{
        dQ,
        dK,
        dV,
        dO,
        static_cast<int64_t>(seq_len) * n_heads * d_head, // batch_stride
        static_cast<int64_t>(n_heads) * d_head,           // seq_stride
        static_cast<int64_t>(d_head),                     // head_stride
        seq_len / Cfg::B_c                                // n_kv_blocks
    };

    const dim3 grid(seq_len / Cfg::B_r, n_heads, batch);
    const dim3 block(Cfg::n_threads);
    auto kernel = MK_NS::flash_forward<value_t>;

    if constexpr (Cfg::smem_bytes > 48 * 1024) {
        CUDA_CHECK(cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
            Cfg::smem_bytes));
    }

    auto launch = [&] { kernel<<<grid, block, Cfg::smem_bytes>>>(args); };

    // ── 1) warmup ───────────────────────────────────────────────────────────
    for (int i = 0; i < warmup; ++i) {
        launch();
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // ── 2) 참고용 시간 측정 (ncu 로 잴 때는 무의미하지만 단독 실행 시 유용) ──
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventRecord(t0));
    for (int i = 0; i < warmup; ++i) {
        launch();
    }
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaDeviceSynchronize());

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, t0, t1));
    const double ms = total_ms / warmup;

    // ── 3) 프로파일 대상: 정확히 1회 ────────────────────────────────────────
    CUDA_CHECK(cudaProfilerStart());
    launch();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaProfilerStop());

    const double flops = 4.0 * batch * n_heads * static_cast<double>(seq_len) *
                         seq_len * d_head;
    std::printf("%-10s  B_r=%-4d B_c=%-4d warps=%d  smem=%2dKB  |  "
                "seq=%-6d b=%-3d h=%-3d  |  %8.3f ms  %7.1f TFLOPs\n",
                MK_STRINGIFY(MK_NS), Cfg::B_r, Cfg::B_c, Cfg::n_warps,
                Cfg::smem_bytes / 1024, seq_len, batch, n_heads, ms,
                flops / (ms * 1e-3) / 1e12);

    CUDA_CHECK(cudaFree(dQ));
    CUDA_CHECK(cudaFree(dK));
    CUDA_CHECK(cudaFree(dV));
    CUDA_CHECK(cudaFree(dO));
    return 0;
}
