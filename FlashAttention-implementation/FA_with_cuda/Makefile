NVCC := nvcc
ARCH := -arch=sm_89
FLAGS := -O3 -std=c++17 $(ARCH)

.PHONY: all clean
all: test_naive test_flash test_flash_tc benchmark benchmark_norm benchmark_tc

test_naive: test_naive.cu common.cuh naive_kernels.cuh
	$(NVCC) $(FLAGS) -o $@ test_naive.cu

test_flash: test_flash.cu common.cuh naive_kernels.cuh flash_kernels.cuh
	$(NVCC) $(FLAGS) -o $@ test_flash.cu

test_flash_tc: test_flash_tc.cu common.cuh flash_kernels.cuh flash_kernels_tc.cuh
	$(NVCC) $(FLAGS) -o $@ test_flash_tc.cu

benchmark: benchmark.cu common.cuh naive_kernels.cuh flash_kernels.cuh
	$(NVCC) $(FLAGS) -o $@ benchmark.cu

benchmark_norm: benchmark_norm.cu common.cuh flash_kernels.cuh
	$(NVCC) $(FLAGS) -o $@ benchmark_norm.cu

benchmark_tc: benchmark_tc.cu common.cuh naive_kernels.cuh flash_kernels.cuh flash_kernels_tc.cuh
	$(NVCC) $(FLAGS) -o $@ benchmark_tc.cu

clean:
	rm -f test_naive test_flash test_flash_tc benchmark benchmark_norm benchmark_tc *.o
