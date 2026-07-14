NVCC        := nvcc
ARCH        := -arch=sm_89
INCLUDES    := -I include
PROFILE_DIR := results/profiles
OMP         := -Xcompiler -fopenmp
LINEINFO	:= -lineinfo

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
	bin/15_gemm_tc_param_tune		\
	bin/16_gemm_tc_swizzle		\
    bin/utils_device_info

.PHONY: all clean run profile info

all: bin/ $(TARGETS)

bin/:
	mkdir -p bin

bin/01_gemm_naive:         src/01_gemm_naive.cu
	$(NVCC) $(LINEINFO) $(ARCH) $(INCLUDES) $(OMP) $< -lcublas -o $@

bin/02_gemm_coalesced:     src/02_gemm_coalesced.cu
	$(NVCC) $(LINEINFO) $(ARCH) $(INCLUDES) $(OMP) $< -lcublas -o $@

bin/03_gemm_shared_memory: src/03_gemm_shared_memory.cu
	$(NVCC) $(LINEINFO) $(ARCH) $(INCLUDES) $(OMP) $< -lcublas -o $@

bin/04_gemm_microtiling:   src/04_gemm_microtiling.cu
	$(NVCC) $(LINEINFO) $(ARCH) $(INCLUDES) $(OMP) $< -lcublas -o $@

bin/05_gemm_vectorization: src/05_gemm_vectorization.cu
	$(NVCC) $(LINEINFO) $(ARCH) $(INCLUDES) $(OMP) $< -lcublas -o $@

bin/06_gemm_param_tune:    src/06_gemm_param_tune.cu
	$(NVCC) $(LINEINFO) $(ARCH) $(INCLUDES) $(OMP) $< -lcublas -o $@

bin/07_gemm_warptiling:    src/07_gemm_warptiling.cu
	$(NVCC) $(LINEINFO) $(ARCH) $(INCLUDES) $(OMP) $< -lcublas -o $@

bin/08_gemm_doublebuffer:  src/08_gemm_doublebuffer.cu
	$(NVCC) $(LINEINFO) $(ARCH) $(INCLUDES) $(OMP) $< -lcublas -o $@

bin/09_gemm_cublas:        src/09_gemm_cublas.cu
	$(NVCC) $(LINEINFO) $(ARCH) $(INCLUDES) $(OMP) $< -lcublas -o $@

bin/utils_device_info:     src/utils_device_info.cu
	$(NVCC) $(LINEINFO) $(ARCH) $(INCLUDES) $< -o $@

# ─── 전체 벤치마크 실행 ──────────────────────────────────────
run: all
	@bash scripts/benchmark.sh

# ─── GPU 정보 출력 ───────────────────────────────────────────
info: bin/utils_device_info
	@./bin/utils_device_info

# ─── NCU 프로파일링 (바이너리가 새로 빌드된 것만 재실행) ──────────
# .ncu-rep 파일마다 대응하는 bin/*을 prerequisite로 걸어서, 소스가 바뀌어
# bin/*이 재빌드된 경우에만 make가 해당 ncu를 다시 돌린다.
PROFILES := \
    $(PROFILE_DIR)/01_naive.ncu-rep           \
    $(PROFILE_DIR)/02_coalesced.ncu-rep       \
    $(PROFILE_DIR)/03_shared_memory.ncu-rep   \
    $(PROFILE_DIR)/04_microtiling.ncu-rep     \
    $(PROFILE_DIR)/05_vectorization.ncu-rep   \
    $(PROFILE_DIR)/06_param_tune.ncu-rep      \
    $(PROFILE_DIR)/07_warptiling.ncu-rep      \
    $(PROFILE_DIR)/08_doublebuffer.ncu-rep    \
    $(PROFILE_DIR)/09_cublas.ncu-rep          \
    $(PROFILE_DIR)/10_tc_naive.ncu-rep        \
    $(PROFILE_DIR)/11_tc_shared_mem.ncu-rep   \
    $(PROFILE_DIR)/12_tc_warptiling.ncu-rep   \
    $(PROFILE_DIR)/13_tc_doublebuffer.ncu-rep \
    $(PROFILE_DIR)/14_tc_vectorization.ncu-rep \
    $(PROFILE_DIR)/15_tc_param_tune.ncu-rep   \
    $(PROFILE_DIR)/16_tc_swizzle.ncu-rep

profile: $(PROFILES)

$(PROFILE_DIR):
	mkdir -p $(PROFILE_DIR)

$(PROFILE_DIR)/01_naive.ncu-rep: bin/01_gemm_naive | $(PROFILE_DIR)
	ncu -f --set full --launch-skip 5 --launch-count 1 -o $(PROFILE_DIR)/01_naive ./bin/01_gemm_naive

$(PROFILE_DIR)/02_coalesced.ncu-rep: bin/02_gemm_coalesced | $(PROFILE_DIR)
	ncu -f --set full --launch-skip 5 --launch-count 1 -o $(PROFILE_DIR)/02_coalesced ./bin/02_gemm_coalesced

$(PROFILE_DIR)/03_shared_memory.ncu-rep: bin/03_gemm_shared_memory | $(PROFILE_DIR)
	ncu -f --set full --launch-skip 5 --launch-count 1 -o $(PROFILE_DIR)/03_shared_memory ./bin/03_gemm_shared_memory

$(PROFILE_DIR)/04_microtiling.ncu-rep: bin/04_gemm_microtiling | $(PROFILE_DIR)
	ncu -f --set full --launch-skip 5 --launch-count 1 -o $(PROFILE_DIR)/04_microtiling ./bin/04_gemm_microtiling

$(PROFILE_DIR)/05_vectorization.ncu-rep: bin/05_gemm_vectorization | $(PROFILE_DIR)
	ncu -f --set full --launch-skip 5 --launch-count 1 -o $(PROFILE_DIR)/05_vectorization ./bin/05_gemm_vectorization

$(PROFILE_DIR)/06_param_tune.ncu-rep: bin/06_gemm_param_tune | $(PROFILE_DIR)
	ncu -f --set full --launch-skip 5 --launch-count 1 -o $(PROFILE_DIR)/06_param_tune ./bin/06_gemm_param_tune

$(PROFILE_DIR)/07_warptiling.ncu-rep: bin/07_gemm_warptiling | $(PROFILE_DIR)
	ncu -f --set full --launch-skip 5 --launch-count 1 -o $(PROFILE_DIR)/07_warptiling ./bin/07_gemm_warptiling

$(PROFILE_DIR)/08_doublebuffer.ncu-rep: bin/08_gemm_doublebuffer | $(PROFILE_DIR)
	ncu -f --set full --launch-skip 5 --launch-count 1 -o $(PROFILE_DIR)/08_doublebuffer ./bin/08_gemm_doublebuffer

$(PROFILE_DIR)/09_cublas.ncu-rep: bin/09_gemm_cublas | $(PROFILE_DIR)
	ncu -f --set full --launch-skip 5 --launch-count 1 -o $(PROFILE_DIR)/09_cublas ./bin/09_gemm_cublas

$(PROFILE_DIR)/10_tc_naive.ncu-rep: bin/10_gemm_tc_naive | $(PROFILE_DIR)
	ncu -f --set full --launch-skip 5 --launch-count 1 -o $(PROFILE_DIR)/10_tc_naive ./bin/10_gemm_tc_naive

$(PROFILE_DIR)/11_tc_shared_mem.ncu-rep: bin/11_gemm_tc_shared_memory | $(PROFILE_DIR)
	ncu -f --set full --launch-skip 5 --launch-count 1 -o $(PROFILE_DIR)/11_tc_shared_mem ./bin/11_gemm_tc_shared_memory

$(PROFILE_DIR)/12_tc_warptiling.ncu-rep: bin/12_gemm_tc_warptiling | $(PROFILE_DIR)
	ncu -f --set full --launch-skip 5 --launch-count 1 -o $(PROFILE_DIR)/12_tc_warptiling ./bin/12_gemm_tc_warptiling

$(PROFILE_DIR)/13_tc_doublebuffer.ncu-rep: bin/13_gemm_tc_doublebuffer | $(PROFILE_DIR)
	ncu -f --set full --launch-skip 5 --launch-count 1 -o $(PROFILE_DIR)/13_tc_doublebuffer ./bin/13_gemm_tc_doublebuffer

$(PROFILE_DIR)/14_tc_vectorization.ncu-rep: bin/14_gemm_tc_vectorization | $(PROFILE_DIR)
	ncu -f --set full --launch-skip 5 --launch-count 1 -o $(PROFILE_DIR)/14_tc_vectorization ./bin/14_gemm_tc_vectorization

$(PROFILE_DIR)/15_tc_param_tune.ncu-rep: bin/15_gemm_tc_param_tune | $(PROFILE_DIR)
	ncu -f --set full --launch-skip 5 --launch-count 1 -o $(PROFILE_DIR)/15_tc_param_tune ./bin/15_gemm_tc_param_tune

$(PROFILE_DIR)/16_tc_swizzle.ncu-rep: bin/16_gemm_tc_swizzle | $(PROFILE_DIR)
	ncu -f --set full --launch-skip 5 --launch-count 1 -o $(PROFILE_DIR)/16_tc_swizzle ./bin/16_gemm_tc_swizzle
# ─── 정리 ────────────────────────────────────────────────────
clean:
	rm -rf bin/
	rm -rf $(PROFILE_DIR)


# ─── Tensor Core 타겟 ────────────────────────────────────────
bin/10_gemm_tc_naive:        src/10_gemm_tc_naive.cu
	$(NVCC) $(LINEINFO) $(ARCH) $(INCLUDES) $(OMP) $< -lcublas -o $@

bin/11_gemm_tc_shared_memory: src/11_gemm_tc_shared_memory.cu
	$(NVCC) $(LINEINFO) $(ARCH) $(INCLUDES) $(OMP) $< -lcublas -o $@

bin/12_gemm_tc_warptiling:   src/12_gemm_tc_warptiling.cu
	$(NVCC) $(LINEINFO) $(ARCH) $(INCLUDES) $(OMP) $< -lcublas -o $@

bin/13_gemm_tc_doublebuffer: src/13_gemm_tc_doublebuffer.cu
	$(NVCC) $(LINEINFO) $(ARCH) $(INCLUDES) $(OMP) $< -lcublas -o $@

bin/14_gemm_tc_vectorization: src/14_gemm_tc_vectorization.cu
	$(NVCC) $(LINEINFO) $(ARCH) $(INCLUDES) $(OMP) $< -lcublas -o $@

bin/15_gemm_tc_param_tune: src/15_gemm_tc_param_tune.cu
	$(NVCC) $(LINEINFO) $(ARCH) $(INCLUDES) $(OMP) $< -lcublas -o $@

bin/16_gemm_tc_swizzle: src/16_gemm_tc_swizzle.cu
	$(NVCC) $(LINEINFO) $(ARCH) $(INCLUDES) $(OMP) $< -lcublas -o $@