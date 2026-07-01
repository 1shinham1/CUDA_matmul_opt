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
    "13|TC Double Buffer |bin/13_gemm_tc_doublebuffer"
    "14|TC Vectorization |bin/14_gemm_tc_vectorization"
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

# ─── cuBLAS FP32 기준값 ───────────────────────────────────────
CUBLAS_OUT=$(./bin/09_gemm_cublas)
CUBLAS_MS=$(echo "$CUBLAS_OUT"     | grep "Average kernel" | awk '{print $5}')
CUBLAS_TFLOPS=$(echo "$CUBLAS_OUT" | grep "GFLOPS:"        | awk '{printf "%.2f", $2/1000}')

echo ""
echo "── CUDA Core (FP32) ──────────────────────────────────────"
echo "[*] cuBLAS FP32 baseline: ${CUBLAS_MS} ms  |  ${CUBLAS_TFLOPS} TFLOPS"
echo "----------------------------------------------------------"

for entry in "${CUDA_KERNELS[@]}"; do
    NUM=$(echo "$entry"  | cut -d'|' -f1)
    NAME=$(echo "$entry" | cut -d'|' -f2)
    CMD=$(echo "$entry"  | cut -d'|' -f3)

    OUTPUT=$(${CMD})
    MS=$(echo "$OUTPUT"     | grep "^Time:"   | awk '{print $2}')
    TFLOPS=$(echo "$OUTPUT" | grep "^TFLOPS:" | awk '{print $2}')
    PCT=$(awk "BEGIN{printf \"%.1f\", ${CUBLAS_MS}/${MS}*100}")

    printf "[%s] %-20s  %6s ms  %5s TFLOPS  (cuBLAS 대비 %s%%)\n" \
        "$NUM" "$NAME" "$MS" "$TFLOPS" "$PCT"
    echo "${NUM}_${NAME// /},${MS},${TFLOPS},${PCT}" >> "${CSV}"
done

printf "[09] %-20s  %6s ms  %5s TFLOPS  (cuBLAS 대비 100.0%%)\n" \
    "cuBLAS FP32" "$CUBLAS_MS" "$CUBLAS_TFLOPS"
echo "09_cuBLAS,${CUBLAS_MS},${CUBLAS_TFLOPS},100.0" >> "${CSV}"

# ─── cuBLAS TF32 기준값 ───────────────────────────────────────
# TC 파일들은 내부에서 cuBLAS TF32를 직접 측정하므로 별도 기준값 추출
TC_REF_OUT=$(./bin/10_gemm_tc_naive 2>/dev/null)
CUBLAS_TC_MS=$(echo "$TC_REF_OUT"     | grep "\[cuBLAS TF32\]" | grep -oP 'time = \K[0-9.]+')
CUBLAS_TC_GFLOPS=$(echo "$TC_REF_OUT" | grep "\[cuBLAS TF32\]" | grep -oP 'GFLOPS = \K[0-9.]+')
CUBLAS_TC_TFLOPS=$(awk "BEGIN{printf \"%.2f\", ${CUBLAS_TC_GFLOPS}/1000}")

echo ""
echo "── Tensor Core (TF32) ────────────────────────────────────"
echo "[*] cuBLAS TF32 baseline: ${CUBLAS_TC_MS} ms  |  ${CUBLAS_TC_TFLOPS} TFLOPS"
echo "----------------------------------------------------------"

for entry in "${TC_KERNELS[@]}"; do
    NUM=$(echo "$entry"  | cut -d'|' -f1)
    NAME=$(echo "$entry" | cut -d'|' -f2)
    CMD=$(echo "$entry"  | cut -d'|' -f3)

    OUTPUT=$(${CMD})
    MS=$(echo "$OUTPUT"     | grep "time = "   | head -1 | grep -oP 'time = \K[0-9.]+')
    GFLOPS=$(echo "$OUTPUT" | grep "GFLOPS = " | head -1 | grep -oP 'GFLOPS = \K[0-9.]+')
    TFLOPS=$(awk "BEGIN{printf \"%.2f\", ${GFLOPS}/1000}")
    PCT=$(awk "BEGIN{printf \"%.1f\", ${CUBLAS_TC_MS}/${MS}*100}")

    printf "[%s] %-20s  %6s ms  %5s TFLOPS  (cuBLAS TF32 대비 %s%%)\n" \
        "$NUM" "$NAME" "$MS" "$TFLOPS" "$PCT"
    echo "${NUM}_${NAME// /},${MS},${TFLOPS},${PCT}" >> "${CSV}"
done

printf "[TC] %-20s  %6s ms  %5s TFLOPS  (cuBLAS TF32 대비 100.0%%)\n" \
    "cuBLAS TF32" "$CUBLAS_TC_MS" "$CUBLAS_TC_TFLOPS"
echo "TC_cuBLAS_TF32,${CUBLAS_TC_MS},${CUBLAS_TC_TFLOPS},100.0" >> "${CSV}"

echo "=========================================================="
echo ""
echo "결과 저장: ${CSV}"