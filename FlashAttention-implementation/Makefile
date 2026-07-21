NVCC ?= nvcc
ARCH ?= sm_89

ROOT_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
BIN_DIR := $(ROOT_DIR)bin
INCLUDE_DIR := $(ROOT_DIR)include
TEST_DIR := $(ROOT_DIR)tests
SRC_DIR := $(ROOT_DIR)src

NVCCFLAGS := -O3 -std=c++17 -arch=$(ARCH) -I$(INCLUDE_DIR)

TEST_BINS := \
	$(BIN_DIR)/test_naive \
	$(BIN_DIR)/test_flash_fma \
	$(BIN_DIR)/test_flash_wmma

BENCH_BINS := \
	$(BIN_DIR)/benchmark_algorithm \
	$(BIN_DIR)/benchmark_normalization \
	$(BIN_DIR)/benchmark_tensor_core \
	$(BIN_DIR)/benchmark_paper_grid

.PHONY: all tests benchmarks test run clean \
	test_naive test_flash test_flash_tc benchmark benchmark_norm benchmark_tc benchmark_paper_grid

all: tests benchmarks

tests: $(TEST_BINS)

benchmarks: $(BENCH_BINS)

$(BIN_DIR):
	mkdir -p $@

$(BIN_DIR)/test_naive: $(TEST_DIR)/test_naive.cu $(INCLUDE_DIR)/common.cuh $(INCLUDE_DIR)/naive_kernels.cuh | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

$(BIN_DIR)/test_flash_fma: $(TEST_DIR)/test_flash.cu $(INCLUDE_DIR)/common.cuh $(INCLUDE_DIR)/naive_kernels.cuh $(INCLUDE_DIR)/flash_kernels.cuh | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

$(BIN_DIR)/test_flash_wmma: $(TEST_DIR)/test_flash_tc.cu $(INCLUDE_DIR)/common.cuh $(INCLUDE_DIR)/flash_kernels.cuh $(INCLUDE_DIR)/flash_kernels_tc.cuh | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

$(BIN_DIR)/benchmark_algorithm: $(SRC_DIR)/benchmark_algorithm.cu $(INCLUDE_DIR)/common.cuh $(INCLUDE_DIR)/naive_kernels.cuh $(INCLUDE_DIR)/flash_kernels.cuh | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

$(BIN_DIR)/benchmark_normalization: $(SRC_DIR)/benchmark_normalization.cu $(INCLUDE_DIR)/common.cuh $(INCLUDE_DIR)/flash_kernels.cuh | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

$(BIN_DIR)/benchmark_tensor_core: $(SRC_DIR)/benchmark_tensor_core.cu $(INCLUDE_DIR)/common.cuh $(INCLUDE_DIR)/naive_kernels.cuh $(INCLUDE_DIR)/flash_kernels.cuh $(INCLUDE_DIR)/flash_kernels_tc.cuh | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

$(BIN_DIR)/benchmark_paper_grid: $(SRC_DIR)/benchmark_paper_grid.cu $(INCLUDE_DIR)/common.cuh $(INCLUDE_DIR)/flash_kernels.cuh $(INCLUDE_DIR)/flash_kernels_tc.cuh | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

# Compatibility aliases for the old target names.
test_naive: $(BIN_DIR)/test_naive
test_flash: $(BIN_DIR)/test_flash_fma
test_flash_tc: $(BIN_DIR)/test_flash_wmma
benchmark: $(BIN_DIR)/benchmark_algorithm
benchmark_norm: $(BIN_DIR)/benchmark_normalization
benchmark_tc: $(BIN_DIR)/benchmark_tensor_core
benchmark_paper_grid: $(BIN_DIR)/benchmark_paper_grid

test: tests
	$(BIN_DIR)/test_naive
	$(BIN_DIR)/test_flash_fma
	$(BIN_DIR)/test_flash_wmma

run: benchmarks
	$(ROOT_DIR)scripts/benchmark.sh --skip-build

clean:
	rm -rf $(BIN_DIR)
