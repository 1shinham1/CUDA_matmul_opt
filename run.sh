#!/usr/bin/env bash
# 사용법:  ./run.sh       → 전체 벤치마크
#          ./run.sh 5     → 05_gemm_vectorization 만 실행
#          ./run.sh 12    → 12_gemm_tc_warptiling 만 실행
 
if [ -z "$1" ]; then
    make run
else
    N=$(printf "%02d" "$1")
    BIN=$(ls bin/${N}_* 2>/dev/null | head -1)
    if [ -z "$BIN" ]; then
        echo "bin/${N}_* 를 찾을 수 없습니다. 먼저 make all 을 실행하세요."
        exit 1
    fi
    echo "▶ $BIN"
    ./$BIN
fi