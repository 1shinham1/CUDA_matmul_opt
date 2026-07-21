#!/usr/bin/env bash
set -e

# ─── CUDA Core 커널 (출력: "Time: X ms", "TFLOPS: Y") ────────
CUDA_KERNELS=(
    "01|Naive            |bin/01_gemm_naive"
    "02|Coalesced        |bin/02_gemm_coalesced"
    "03|Shared Memory    |bin/03_gemm_shared_memory"
    "04|Microtiling      |bin/04_gemm_microtiling"
    "05|Vectorization    |bin/05_gemm_vectorization"
    "06|Parameter Tuning |bin/06_gemm_param_tune"
    "07|Warp Tiling      |bin/07_gemm_warptiling"
    "08|Double Buffering |bin/08_gemm_doublebuffer"
)

# ─── Tensor Core 커널 (출력: "time = X ms", "GFLOPS = Y") ────
TC_KERNELS=(
    "10|TC Naive         |bin/10_gemm_tc_naive"
    "11|TC Shared Memory |bin/11_gemm_tc_shared_memory"
    "12|TC Warp Tiling   |bin/12_gemm_tc_warptiling"
    "13|TC Vectorization |bin/13_gemm_tc_vectorization"
    "14|TC Vectorization+DB|bin/14_gemm_tc_doublebuffer"
    "15|TC Param Tune    |bin/15_gemm_tc_param_tune"
    "16|TC Swizzle       |bin/16_gemm_tc_swizzle"
)

RESULTS_DIR="results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
CSV="${RESULTS_DIR}/benchmark_${TIMESTAMP}.csv"
mkdir -p "${RESULTS_DIR}"

echo "kernel,time_ms,tflops,cublas_pct" > "${CSV}"

echo "=========================================================="
echo "  CUDA GEMM Benchmark  (4096×4096×4096, FP32, sm_89)"
echo "  $(date)"
echo "=========================================================="

# ─── CUDA Core (FP32) ────────────────────────────────────────
# 01~08도 TC 커널과 동일하게 자기 프로세스 안에서 커널 직후 cuBLAS FP32를
# 바로 이어서 측정해 출력한다(run_cublas_fp32_and_verify). 09번(cuBLAS 자체
# 실행)의 측정값을 01~08에 공유하면 프로세스 간 GPU 클럭/부스트 편차가 %에
# 섞이므로, 커널마다 "자체 측정값"으로 %를 계산한다.
echo ""
echo "── CUDA Core (FP32) ──────────────────────────────────────"
echo "----------------------------------------------------------"

for entry in "${CUDA_KERNELS[@]}"; do
    NUM=$(echo "$entry"  | cut -d'|' -f1)
    NAME=$(echo "$entry" | cut -d'|' -f2)
    CMD=$(echo "$entry"  | cut -d'|' -f3)

    OUTPUT=$(${CMD})
    MS=$(echo "$OUTPUT"     | grep "^Time:"   | awk '{print $2}')
    TFLOPS=$(echo "$OUTPUT" | grep "^TFLOPS:" | awk '{print $2}')

    CUBLAS_OWN_MS=$(echo "$OUTPUT" | grep "\[cuBLAS FP32\]" | grep -oP 'time = \K[0-9.]+')
    PCT=$(awk "BEGIN{printf \"%.1f\", ${CUBLAS_OWN_MS}/${MS}*100}")

    printf "[%s] %-20s  %6s ms  %5s TFLOPS  (cuBLAS FP32 %sms 대비 %s%%)\n" \
        "$NUM" "$NAME" "$MS" "$TFLOPS" "$CUBLAS_OWN_MS" "$PCT"
    echo "${NUM}_${NAME// /},${MS},${TFLOPS},${PCT}" >> "${CSV}"
done

CUBLAS_OUT=$(./bin/09_gemm_cublas)
CUBLAS_MS=$(echo "$CUBLAS_OUT"     | grep "Average kernel" | awk '{print $5}')
CUBLAS_TFLOPS=$(echo "$CUBLAS_OUT" | grep "GFLOPS:"        | awk '{printf "%.2f", $2/1000}')
printf "[09] %-20s  %6s ms  %5s TFLOPS  (cuBLAS 대비 100.0%%)\n" \
    "cuBLAS FP32" "$CUBLAS_MS" "$CUBLAS_TFLOPS"
echo "09_cuBLAS,${CUBLAS_MS},${CUBLAS_TFLOPS},100.0" >> "${CSV}"

# ─── Tensor Core (TF32) ─────────────────────────────────────
# TC 파일들은 자기 프로세스 안에서 자기 커널 직후 cuBLAS TF32를 바로 이어서
# 측정해 출력한다(run_cublas_and_verify). 커널마다 그 "자체 측정값"으로 %를
# 계산해야 프로세스 간 GPU 클럭/부스트 편차가 안 섞인다. (10번 커널 한 번
# 측정값을 11~15번에 공유해서 재사용하면 그 편차만큼 %가 왜곡됐었음)
echo ""
echo "── Tensor Core (TF32) ────────────────────────────────────"
echo "----------------------------------------------------------"

for entry in "${TC_KERNELS[@]}"; do
    NUM=$(echo "$entry"  | cut -d'|' -f1)
    NAME=$(echo "$entry" | cut -d'|' -f2)
    CMD=$(echo "$entry"  | cut -d'|' -f3)

    OUTPUT=$(${CMD})
    MS=$(echo "$OUTPUT"     | grep "time = "   | head -1 | grep -oP 'time = \K[0-9.]+')
    GFLOPS=$(echo "$OUTPUT" | grep "GFLOPS = " | head -1 | grep -oP 'GFLOPS = \K[0-9.]+')
    TFLOPS=$(awk "BEGIN{printf \"%.2f\", ${GFLOPS}/1000}")

    CUBLAS_OWN_MS=$(echo "$OUTPUT" | grep "\[cuBLAS TF32\]" | grep -oP 'time = \K[0-9.]+')
    PCT=$(awk "BEGIN{printf \"%.1f\", ${CUBLAS_OWN_MS}/${MS}*100}")

    printf "[%s] %-20s  %6s ms  %5s TFLOPS  (cuBLAS TF32 %sms 대비 %s%%)\n" \
        "$NUM" "$NAME" "$MS" "$TFLOPS" "$CUBLAS_OWN_MS" "$PCT"
    echo "${NUM}_${NAME// /},${MS},${TFLOPS},${PCT}" >> "${CSV}"
done

echo "=========================================================="
echo ""
echo "결과 저장: ${CSV}"