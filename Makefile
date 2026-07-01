NVCC        := nvcc
ARCH        := -arch=sm_89
INCLUDES    := -I include
PROFILE_DIR := results/profiles

# ─── 빌드 타겟 ───────────────────────────────────────────────
TARGETS := \
    bin/01_gemm_naive       \
    bin/02_gemm_coalesced   \
    bin/03_gemm_shared_memory \
    bin/04_gemm_microtiling \
    bin/05_gemm_vectorization \
    bin/06_gemm_param_tune  \
    bin/07_gemm_warptiling  \
    bin/08_gemm_doublebuffer \
    bin/09_gemm_cublas      \
    bin/10_gemm_tc_naive      \
    bin/11_gemm_tc_shared_memory \
    bin/12_gemm_tc_warptiling \
    bin/13_gemm_tc_doublebuffer \
	bin/14_gemm_tc_vectorization \
    bin/utils_device_info

.PHONY: all clean run profile info

all: bin/ $(TARGETS)

bin/:
	mkdir -p bin

bin/01_gemm_naive:         src/01_gemm_naive.cu
	$(NVCC) $(ARCH) $(INCLUDES) $< -o $@

bin/02_gemm_coalesced:     src/02_gemm_coalesced.cu
	$(NVCC) $(ARCH) $(INCLUDES) $< -o $@

bin/03_gemm_shared_memory: src/03_gemm_shared_memory.cu
	$(NVCC) $(ARCH) $(INCLUDES) $< -o $@

bin/04_gemm_microtiling:   src/04_gemm_microtiling.cu
	$(NVCC) $(ARCH) $(INCLUDES) $< -o $@

bin/05_gemm_vectorization: src/05_gemm_vectorization.cu
	$(NVCC) $(ARCH) $(INCLUDES) $< -o $@

bin/06_gemm_param_tune:    src/06_gemm_param_tune.cu
	$(NVCC) $(ARCH) $(INCLUDES) $< -o $@

bin/07_gemm_warptiling:    src/07_gemm_warptiling.cu
	$(NVCC) $(ARCH) $(INCLUDES) $< -o $@

bin/08_gemm_doublebuffer:  src/08_gemm_doublebuffer.cu
	$(NVCC) $(ARCH) $(INCLUDES) $< -o $@

bin/09_gemm_cublas:        src/09_gemm_cublas.cu
	$(NVCC) $(ARCH) $(INCLUDES) $< -lcublas -o $@

bin/utils_device_info:     src/utils_device_info.cu
	$(NVCC) $(ARCH) $(INCLUDES) $< -o $@

# ─── 전체 벤치마크 실행 ──────────────────────────────────────
run: all
	@bash scripts/benchmark.sh

# ─── GPU 정보 출력 ───────────────────────────────────────────
info: bin/utils_device_info
	@./bin/utils_device_info

# ─── NCU 프로파일링 ──────────────────────────────────────────
profile: all
	@mkdir -p $(PROFILE_DIR)
	ncu -f --set full --launch-skip 1 --launch-count 1 -o $(PROFILE_DIR)/01_naive           ./bin/01_gemm_naive
	ncu -f --set full --launch-skip 1 --launch-count 1 -o $(PROFILE_DIR)/02_coalesced       ./bin/02_gemm_coalesced
	ncu -f --set full --launch-skip 1 --launch-count 1 -o $(PROFILE_DIR)/03_shared_memory   ./bin/03_gemm_shared_memory
	ncu -f --set full --launch-skip 1 --launch-count 1 -o $(PROFILE_DIR)/04_microtiling     ./bin/04_gemm_microtiling
	ncu -f --set full --launch-skip 1 --launch-count 1 -o $(PROFILE_DIR)/05_vectorization   ./bin/05_gemm_vectorization
	ncu -f --set full --launch-skip 1 --launch-count 1 -o $(PROFILE_DIR)/06_param_tune      ./bin/06_gemm_param_tune
	ncu -f --set full --launch-skip 1 --launch-count 1 -o $(PROFILE_DIR)/07_warptiling      ./bin/07_gemm_warptiling
	ncu -f --set full --launch-skip 1 --launch-count 1 -o $(PROFILE_DIR)/08_doublebuffer    ./bin/08_gemm_doublebuffer
	ncu -f --set full --launch-skip 10 --launch-count 1 -o $(PROFILE_DIR)/09_cublas         ./bin/09_gemm_cublas
	ncu -f --set full --launch-skip 1 --launch-count 1 -o $(PROFILE_DIR)/10_tc_naive        ./bin/10_gemm_tc_naive
	ncu -f --set full --launch-skip 1 --launch-count 1 -o $(PROFILE_DIR)/11_tc_shared_mem   ./bin/11_gemm_tc_shared_memory
	ncu -f --set full --launch-skip 1 --launch-count 1 -o $(PROFILE_DIR)/12_tc_warptiling   ./bin/12_gemm_tc_warptiling
	ncu -f --set full --launch-skip 1 --launch-count 1 -o $(PROFILE_DIR)/13_tc_doublebuffer ./bin/13_gemm_tc_doublebuffer
	ncu -f --set full --launch-skip 1 --launch-count 1 -o $(PROFILE_DIR)/14_tc_vectorization ./bin/14_gemm_tc_vectorization

# ─── 정리 ────────────────────────────────────────────────────
clean:
	rm -rf bin/
	rm -rf $(PROFILE_DIR)


# ─── Tensor Core 타겟 ────────────────────────────────────────
bin/10_gemm_tc_naive:        src/10_gemm_tc_naive.cu
	$(NVCC) $(ARCH) $(INCLUDES) $< -lcublas -o $@

bin/11_gemm_tc_shared_memory: src/11_gemm_tc_shared_memory.cu
	$(NVCC) $(ARCH) $(INCLUDES) $< -lcublas -o $@

bin/12_gemm_tc_warptiling:   src/12_gemm_tc_warptiling.cu
	$(NVCC) $(ARCH) $(INCLUDES) $< -lcublas -o $@

bin/13_gemm_tc_doublebuffer: src/13_gemm_tc_doublebuffer.cu
	$(NVCC) $(ARCH) $(INCLUDES) $< -lcublas -o $@

bin/14_gemm_tc_vectorization: src/14_gemm_tc_vectorization.cu
	$(NVCC) $(ARCH) $(INCLUDES) $< -lcublas -o $@