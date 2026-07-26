#!/usr/bin/env bash
# 사용법:  ./run.sh        → 전체 빌드 + 벤치마크 (make run)
#          ./run.sh 1      → 01_flash_tc_naive 만 실행
#          ./run.sh 1 causal → 01_flash_tc_naive 를 causal 모드로 실행
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_dir"

if [ -z "${1:-}" ]; then
    make run
else
    N=$(printf "%02d" "$1")
    BIN=$(ls bin/${N}_* 2>/dev/null | head -1)
    if [ -z "$BIN" ]; then
        echo "bin/${N}_* 를 찾을 수 없습니다. 먼저 make all 을 실행하세요."
        exit 1
    fi
    echo "▶ $BIN ${2:-}"
    ./$BIN ${2:-}
fi
