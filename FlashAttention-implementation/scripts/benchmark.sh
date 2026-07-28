#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
results_dir="$project_dir/results"

if [[ "${1:-}" != "--skip-build" ]]; then
    make -C "$project_dir" all
fi

mkdir -p "$results_dir"
cd "$project_dir"

BINARIES=(
    bin/00_attention_naive
    bin/01_flash_tc_naive
    bin/02_flash_tc_async
)

for bin in "${BINARIES[@]}"; do
    echo "=========================================================="
    echo " $bin (non-causal)"
    echo "=========================================================="
    "$bin"
    echo
    echo "=========================================================="
    echo " $bin causal"
    echo "=========================================================="
    "$bin" causal
    echo
done

timestamp="$(date +%Y%m%d_%H%M%S)"
snapshot_dir="$results_dir/runs/$timestamp"
mkdir -p "$snapshot_dir"
cp "$results_dir"/results_*.csv "$snapshot_dir"/ 2>/dev/null || true
printf 'Benchmark snapshot: %s\n' "$snapshot_dir"
