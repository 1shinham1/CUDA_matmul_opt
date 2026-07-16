// [벤치마크] naive attention vs FlashAttention CUDA 커널 성능 비교
//
// 시퀀스 길이(seq_len)를 늘려가면서 "forward만" / "forward+backward"
// 두 가지 경우로 실행 시간(ms)과 GPU 메모리 사용량(MB)을 측정한다.
// batch*heads=32, head_dim=64 설정으로 측정 (compare/ 디렉토리에서
// FA_official과 head-to-head 비교할 때 쓰는 것과 같은 설정).
//
// 중요한 설계 포인트: S, P, dP, dS 같은 스크래치(임시) 버퍼들은
// 시간을 측정하는 구간 "밖에서" 미리 한 번만 할당한다. 만약 매
// 반복(iteration)마다 cudaMalloc/cudaFree를 하면, 우리가 측정하려는
// "커널 연산 자체의 속도"가 아니라 "메모리 할당 오버헤드"까지
// 같이 측정되어 버려서 공정한 비교가 안 된다.
#include "naive_kernels.cuh"
#include "flash_kernels.cuh"
#include <cstdio>
#include <fstream>
#include <string>

// HEAD_DIM과 FLASH_BLOCK_N은 flash_kernels.cuh에서 템플릿 매개변수로
// 쓰이기 때문에 "컴파일 타임 상수"여야 한다 (실행 중에 바꿀 수 없음).
// 반면 FLASH_BLOCK_M(Q 타일 크기)은 커널 launch 시 blockDim.x로
// 넘겨주는 "실행 시점 값"이라서 굳이 상수일 필요는 없지만, 편의상
// 여기서는 그냥 상수로 고정해둔 것.
constexpr int HEAD_DIM = 64;
constexpr int FLASH_BLOCK_N = 32;
constexpr int FLASH_BLOCK_M = 64;

// cudaMalloc을 try/catch처럼 "일단 해보고 실패하면 처리"하는 대신,
// 미리 cudaMemGetInfo()로 "지금 이만큼 남은 메모리에 이 할당이
// 들어갈 수 있는가?"를 먼저 확인하는 함수. 1.2배 여유를 두는 이유는
// CUDA 컨텍스트 자체가 차지하는 메모리, 그리고 이 GPU를 같이 쓰고
// 있는 다른 프로세스(nvidia-smi에 보이던 수백MB짜리 프로세스)까지
// 감안해서 안전 마진을 주기 위함. 마진이 없으면 "이론상 딱 맞아서
// fits_in_gpu()는 true인데 실제로는 cudaMalloc이 실패"하는 경우가
// 생길 수 있다.
static bool fits_in_gpu(size_t bytes_needed) {
    size_t free_b, total_b;
    cudaMemGetInfo(&free_b, &total_b);
    return (double)bytes_needed * 1.2 < (double)free_b;
}

// seq_len(시퀀스 길이)별로 몇 번 반복 측정해서 중앙값(median)을 낼지 결정.
// - seq_len이 작으면: 커널 한 번 실행이 마이크로초 단위로 끝나기 때문에,
//   GPU 스케줄링 지터(jitter)에 흔들리지 않는 안정적인 값을 얻으려면
//   반복 횟수를 늘려야 한다.
// - seq_len이 크면: 한 번 실행 자체가 이미 충분히 느려서 반복을 많이 안 해도
//   안정적이고, 전체 벤치마크가 너무 오래 걸리지 않게 반복을 줄인다.
// common.cuh의 time_ms_median()이 이 반복들 "이전에" 3번의 워밍업
// 실행을 추가로 하는데, 이건 CUDA 컨텍스트 초기화나 첫 커널 launch 시의
// 일회성 비용(예: cudaFuncSetAttribute 최초 호출)을 측정에서 빼기 위함.
static int iters_for(int seq_len) {
    if (seq_len <= 1024) return 20;
    if (seq_len <= 4096) return 10;
    return 5;
}

// 시퀀스 길이 seq_len 하나에 대한 결과 한 줄. "fwd"는 forward만 돌린 경우,
// "fb"는 forward+backward를 같이 돌린 경우(실제 학습 상황과 더
// 비슷한 수치)를 의미한다. 기본값 -1은 "GPU 메모리가 부족해서
// 아예 실행을 건너뛰었다"는 표시용 값 -- 표에 빈칸을 남기는 대신
// -1.0000으로 찍혀서, 나중에 표를 볼 때 "측정을 안 한 것"과
// "측정했는데 이 값이 나온 것"을 한눈에 구분할 수 있게 했다.
struct Row {
    int seq_len;
    double naive_fwd_ms = -1, naive_fwd_mb = -1;
    double flash_fwd_ms = -1, flash_fwd_mb = -1;
    double naive_fb_ms = -1, naive_fb_mb = -1;
    double flash_fb_ms = -1, flash_fb_mb = -1;
};

int main(int argc, char** argv) {
    // 커맨드라인 인자로 "causal"을 주면 causal(인과적) 마스킹 모드로 측정.
    // 예: ./benchmark        -> non-causal
    //     ./benchmark causal -> causal (GPT처럼 미래 토큰을 못 보게 마스킹)
    bool causal = argc > 1 && std::string(argv[1]) == "causal";
    int BH = 32;  // batch=4, heads=8 -> batch*heads=32 (compare/의 FA_official 비교와 동일 설정).
    printf("FA_with_cuda benchmark (BH=%d, head_dim=%d, causal=%d, Q/K/V/O/dQ/dK/dV/dO=fp16, compute=fp32)\n\n", BH, HEAD_DIM, causal);

    // 128부터 32768까지 2배씩 늘려가며 측정. naive는 메모리 부족(OOM)으로
    // 중간에 못 돌게 될 것이고(O(seq_len^2) 메모리라서), flash는 O(seq_len)이라
    // 끝까지 돌 것으로 예상 -- 이게 바로 우리가 보여주고 싶은 핵심 결과.
    int seq_lens[] = {128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768};

    // 한 번이라도 "메모리가 안 맞아서 못 돌렸다"고 판단되면, 그 이후의
    // (더 큰) seq_len에 대해서는 다시 시도해봤자 어차피 또 안 되는 게
    // 뻔하므로 아예 건너뛴다 (시간 절약용 플래그).
    bool naive_fwd_feasible = true, naive_fb_feasible = true;
    std::vector<Row> rows;

    for (int seq_len : seq_lens) {
        Row row; row.seq_len = seq_len;
        size_t qkv_elems = (size_t)BH * seq_len * HEAD_DIM;
        size_t qkv_bytes = qkv_elems * sizeof(__half);  // Q,K,V,O,dQ,dK,dV,dO는 fp16 저장이라 sizeof(__half)(=2바이트)
        size_t sp_bytes = (size_t)BH * seq_len * seq_len * sizeof(float);  // S, P, dP, dS는 fp32 유지 (O(seq_len^2)!)
        size_t l_bytes = (size_t)BH * seq_len * sizeof(float);       // logsumexp L, D도 fp32 (O(seq_len))
        printf("seq_len=%d ...\n", seq_len); fflush(stdout);

        // Q, K, V, dO를 CPU(호스트)에서 fp32 랜덤하게 채운 뒤 fp16으로 변환해서 GPU로 복사.
        // 실제 attention 계산 내용 자체는 중요하지 않고(랜덤 값이면 충분),
        // 여기서는 순수하게 "속도/메모리"만 측정하는 것이 목적이라서
        // 정확도 검증은 test_naive.cu / test_flash.cu에서 이미 따로 했다.
        std::vector<float> hQ(qkv_elems), hK(qkv_elems), hV(qkv_elems), hdO(qkv_elems);
        fill_random(hQ, 1); fill_random(hK, 2); fill_random(hV, 3); fill_random(hdO, 4);
        std::vector<__half> hQh = to_half(hQ), hKh = to_half(hK), hVh = to_half(hV), hdOh = to_half(hdO);

        // Q,K,V,O,dO,dQ,dK,dV -- 두 구현(naive/flash) 모두가 공통으로
        // 필요로 하는 입력/출력 버퍼들을 여기서 한 번만 할당하고 이후
        // naive 쪽 측정과 flash 쪽 측정에서 그대로 재사용한다.
        DeviceAllocTracker alloc;
        __half *Q = (__half*)alloc.alloc(qkv_bytes), *K = (__half*)alloc.alloc(qkv_bytes);
        __half *V = (__half*)alloc.alloc(qkv_bytes), *O = (__half*)alloc.alloc(qkv_bytes);
        __half *dO = (__half*)alloc.alloc(qkv_bytes);
        __half *dQ = (__half*)alloc.alloc(qkv_bytes), *dK = (__half*)alloc.alloc(qkv_bytes), *dV = (__half*)alloc.alloc(qkv_bytes);
        CUDA_CHECK(cudaMemcpy(Q, hQh.data(), qkv_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(K, hKh.data(), qkv_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(V, hVh.data(), qkv_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dO, hdOh.data(), qkv_bytes, cudaMemcpyHostToDevice));

        // ==================== naive forward ====================
        // naive는 S, P라는 seq_len x seq_len 크기 버퍼가 필요하다 (naive_kernels.cuh
        // 참고: S=QK^T, P=softmax(S)를 각각 별도 버퍼에 저장하는 방식).
        // 이게 naive의 O(seq_len^2) 메모리 문제의 원인이자, flash와 비교할 때
        // 가장 크게 갈리는 지점이다.
        if (naive_fwd_feasible && fits_in_gpu(2 * sp_bytes)) {
            float* S = (float*)alloc.alloc(sp_bytes);
            float* P = (float*)alloc.alloc(sp_bytes);

            // 람다(run)로 감싸서 time_ms_median에 넘긴다: 이 안에서는
            // 순수하게 커널 launch 3개(scores, softmax, output)만
            // 일어나고, 메모리 할당은 이미 위에서 끝난 상태.
            auto run = [&]() { naive_forward_launch(Q, K, V, O, S, P, BH, seq_len, HEAD_DIM, causal); };
            row.naive_fwd_ms = time_ms_median(run, 3, iters_for(seq_len));
            // peak_mb()는 "지금까지 이 alloc 객체로 할당한 것들의 최댓값"이라
            // Q,K,V,O,dO,dQ,dK,dV + S,P가 전부 포함된 "이 시점까지의 총
            // 사용량"을 나타낸다 (아직 dP,dS는 할당 전이라 안 들어있음).
            row.naive_fwd_mb = alloc.peak_mb();

            // ---- naive forward+backward (S,P는 재사용, dP,dS만 추가 할당) ----
            if (naive_fb_feasible && fits_in_gpu(2 * sp_bytes)) {
                float* dP = (float*)alloc.alloc(sp_bytes);
                float* dS = (float*)alloc.alloc(sp_bytes);
                // reset_peak(): 지금까지의 누적 최대치를 "현재 할당량"
                // 기준으로 리셋. 이렇게 해야 peak_mb()가 "forward+backward를
                // 도는 동안의 메모리"를 제대로 보여준다 (이 구간에서는
                // 추가 할당이 없으므로 사실상 "지금 시점의 총 사용량"과
                // 같은 값이 나오지만, 의미를 명확히 하기 위해 리셋해둠).
                alloc.reset_peak();
                auto run_fb = [&]() {
                    naive_forward_launch(Q, K, V, O, S, P, BH, seq_len, HEAD_DIM, causal);
                    naive_backward_launch(Q, K, V, dO, P, dP, dS, dQ, dK, dV, BH, seq_len, HEAD_DIM, causal);
                };
                row.naive_fb_ms = time_ms_median(run_fb, 3, iters_for(seq_len));
                row.naive_fb_mb = alloc.peak_mb();
                alloc.free(dP, sp_bytes);
                alloc.free(dS, sp_bytes);
            } else {
                // 한 번 못 돌면 이후 seq_len에서는 더 못 돌 게 뻔하므로 플래그를
                // 내려서 다음 반복부터는 아예 시도조차 안 하게 만든다.
                naive_fb_feasible = false;
                printf("    naive fwd+bwd: skipped (would not fit), seq_len=%d\n", seq_len);
            }

            alloc.free(S, sp_bytes);
            alloc.free(P, sp_bytes);
        } else {
            naive_fwd_feasible = false;
            naive_fb_feasible = false;  // forward도 안 되면 backward는 말할 것도 없음
            printf("    naive fwd: skipped (would not fit), seq_len=%d\n", seq_len);
        }

        // ==================== flash forward ====================
        // flash는 seq_len x seq_len짜리 버퍼가 전혀 필요 없다. forward가 필요로 하는
        // 추가 메모리는 로그-합-지수(logsumexp) L 하나뿐이고, 이건
        // BH*seq_len 크기(=O(seq_len))라서 naive의 S,P(=O(seq_len^2))에 비하면 무시할
        // 수준이다. 그래서 naive와 달리 fits_in_gpu 체크도 필요 없이
        // 항상 시도한다 (seq_len=32768까지도 문제없이 들어감).
        {
            float* L = (float*)alloc.alloc(l_bytes);
            alloc.reset_peak();
            auto run = [&]() { flash_forward_launch<HEAD_DIM, FLASH_BLOCK_N>(Q, K, V, O, L, BH, seq_len, causal, FLASH_BLOCK_M); };
            row.flash_fwd_ms = time_ms_median(run, 3, iters_for(seq_len));
            row.flash_fwd_mb = alloc.peak_mb();

            // ---- flash forward+backward (L은 재사용, D만 추가 할당) ----
            // D = rowsum(dO * O) 는 backward에서 필요한 통계량으로, 이것도
            // O(seq_len) 크기라 flash 전체 메모리는 여전히 O(seq_len)을 유지한다.
            float* D = (float*)alloc.alloc(l_bytes);
            alloc.reset_peak();
            auto run_fb = [&]() {
                flash_forward_launch<HEAD_DIM, FLASH_BLOCK_N>(Q, K, V, O, L, BH, seq_len, causal, FLASH_BLOCK_M);
                flash_backward_launch_buf<HEAD_DIM, FLASH_BLOCK_N>(Q, K, V, O, dO, L, D, dQ, dK, dV, BH, seq_len, causal, FLASH_BLOCK_M);
            };
            row.flash_fb_ms = time_ms_median(run_fb, 3, iters_for(seq_len));
            row.flash_fb_mb = alloc.peak_mb();

            alloc.free(D, l_bytes);
            alloc.free(L, l_bytes);
        }

        alloc.free(Q, qkv_bytes); alloc.free(K, qkv_bytes); alloc.free(V, qkv_bytes); alloc.free(O, qkv_bytes);
        alloc.free(dO, qkv_bytes); alloc.free(dQ, qkv_bytes); alloc.free(dK, qkv_bytes); alloc.free(dV, qkv_bytes);
        rows.push_back(row);
    }

    // ==================== 결과 출력 ====================
    // 콘솔에 표로 한 번 찍고, 같은 내용을 CSV로도 저장한다 (causal이면
    // results_causal.csv, 아니면 results_noncausal.csv).
    printf("\n%6s | %10s %10s | %10s %10s | %10s %10s | %10s %10s\n",
           "seq_len", "naive_fwd", "naive_mb", "flash_fwd", "flash_mb", "naive_fb", "naivefb_mb", "flash_fb", "flashfb_mb");
    std::string csv_path = causal ? "results_causal.csv" : "results_noncausal.csv";
    std::ofstream csv(csv_path);
    csv << "seq_len,naive_fwd_ms,naive_fwd_mb,flash_fwd_ms,flash_fwd_mb,naive_fb_ms,naive_fb_mb,flash_fb_ms,flash_fb_mb\n";
    for (auto& r : rows) {
        printf("%6d | %10.4f %10.2f | %10.4f %10.2f | %10.4f %10.2f | %10.4f %10.2f\n",
               r.seq_len, r.naive_fwd_ms, r.naive_fwd_mb, r.flash_fwd_ms, r.flash_fwd_mb,
               r.naive_fb_ms, r.naive_fb_mb, r.flash_fb_ms, r.flash_fb_mb);
        csv << r.seq_len << "," << r.naive_fwd_ms << "," << r.naive_fwd_mb << ","
            << r.flash_fwd_ms << "," << r.flash_fwd_mb << ","
            << r.naive_fb_ms << "," << r.naive_fb_mb << ","
            << r.flash_fb_ms << "," << r.flash_fb_mb << "\n";
    }
    printf("\nSaved %s\n", csv_path.c_str());
    return 0;
}
