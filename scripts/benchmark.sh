#!/usr/bin/env bash
set -e

KERNELS=(
    "01|Naive            |bin/01_gemm_naive"
    "02|Coalesced        |bin/02_gemm_coalesced"
    "03|Shared Memory    |bin/03_gemm_shared_memory"
    "04|Microtiling      |bin/04_gemm_microtiling"
    "05|Vectorization    |bin/05_gemm_vectorization"
    "06|Parameter Tuning |bin/06_gemm_param_tune"
    "07|Warp Tiling      |bin/07_gemm_warptiling"
    "08|Double Buffering |bin/08_gemm_doublebuffer"
)

# 결과 저장 경로
RESULTS_DIR="results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
CSV="${RESULTS_DIR}/benchmark_${TIMESTAMP}.csv"
mkdir -p "${RESULTS_DIR}"

echo "kernel,time_ms,tflops,cublas_pct" > "${CSV}"

echo "=========================================================="
echo "  CUDA GEMM Benchmark  (4096×4096×4096, FP32, sm_89)"
echo "  $(date)"
echo "=========================================================="

# cuBLAS 기준값 측정
CUBLAS_OUT=$(./bin/09_gemm_cublas)
CUBLAS_MS=$(echo "$CUBLAS_OUT"     | grep "Average kernel" | awk '{print $5}')
CUBLAS_TFLOPS=$(echo "$CUBLAS_OUT" | grep "GFLOPS:"        | awk '{printf "%.2f", $2/1000}')

echo ""
echo "[*] cuBLAS baseline: ${CUBLAS_MS} ms  |  ${CUBLAS_TFLOPS} TFLOPS"
echo "----------------------------------------------------------"

for entry in "${KERNELS[@]}"; do
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

echo "----------------------------------------------------------"
printf "[09] %-20s  %6s ms  %5s TFLOPS  (cuBLAS 대비 100.0%%)\n" \
    "cuBLAS" "$CUBLAS_MS" "$CUBLAS_TFLOPS"
echo "=========================================================="
echo ""
echo "결과 저장: ${CSV}"

echo "09_cuBLAS,${CUBLAS_MS},${CUBLAS_TFLOPS},100.0" >> "${CSV}"
