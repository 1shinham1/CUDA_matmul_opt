#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
results_dir="$project_dir/results"

if [[ "${1:-}" != "--skip-build" ]]; then
    make -C "$project_dir" benchmarks
fi

mkdir -p "$results_dir"
cd "$project_dir"

bin/benchmark_algorithm
bin/benchmark_algorithm causal
bin/benchmark_normalization
bin/benchmark_normalization causal
bin/benchmark_tensor_core
bin/benchmark_tensor_core causal
bin/benchmark_paper_grid

timestamp="$(date +%Y%m%d_%H%M%S)"
snapshot_dir="$results_dir/runs/$timestamp"
mkdir -p "$snapshot_dir"
cp "$results_dir"/results_*.csv "$snapshot_dir"/
printf 'Benchmark snapshot: %s\n' "$snapshot_dir"
