NVCC        := nvcc
ARCH        := -arch=sm_89
PROFILE_DIR := ncu_rep_dir

# Sources → Binaries
TARGETS := \
    coalesced \
    cuBLAS \
    DeviceProperties \
    doublebuffer \
    microtiling \
    naive \
    parameterTune \
    tiled \
    vectorization \
    warptiling

# Default: build everything
.PHONY: all clean run

all: $(TARGETS)

coalesced:      coalesced.cu
	$(NVCC) $(ARCH) $< -o $@

cuBLAS:         cuBLAS.cu
	$(NVCC) $(ARCH) $< -lcublas -o $@

DeviceProperties: DeviceProperties.cu
	$(NVCC) $(ARCH) $< -o $@

doublebuffer:   doublebuffer.cu
	$(NVCC) $(ARCH) $< -o $@

microtiling:    microtiling.cu
	$(NVCC) $(ARCH) $< -o $@

naive:          naive.cu
	$(NVCC) $(ARCH) $< -o $@

parameterTune:  parameterTune.cu
	$(NVCC) $(ARCH) $< -o $@

tiled:          tiled.cu
	$(NVCC) $(ARCH) $< -o $@

vectorization:  vectorization.cu
	$(NVCC) $(ARCH) $< -o $@

warptiling:     warptiling.cu
	$(NVCC) $(ARCH) $< -o $@

# Run all benchmarks in optimization order
run: all
	@echo "=========================================="
	@echo " CUDA MatMul Benchmark  (sm_89)"
	@echo "=========================================="
	@CUBLAS_MS=$$(./cuBLAS | grep "Average kernel" | awk '{print $$5}'); \
	for entry in \
	    "1|Naive|./naive" \
	    "2|Coalesced|./coalesced" \
	    "3|Tiled|./tiled" \
	    "4|Microtiling|./microtiling" \
	    "5|Vectorization|./vectorization" \
	    "6|Parameter Tuning|./parameterTune" \
	    "7|Warptiling|./warptiling" \
	    "8|Double Buffering|./doublebuffer" \
	; do \
	    NUM=$$(echo $$entry | cut -d'|' -f1); \
	    NAME=$$(echo $$entry | cut -d'|' -f2); \
	    CMD=$$(echo $$entry | cut -d'|' -f3); \
	    OUTPUT=$$($$CMD); \
	    MS=$$(echo "$$OUTPUT" | grep "^Time:" | awk '{print $$2}'); \
	    PCT=$$(awk "BEGIN{printf \"%.1f\", $$CUBLAS_MS/$$MS*100}"); \
	    echo "[$$NUM] $$NAME"; \
	    echo "$$OUTPUT"; \
	    echo "vs cuBLAS: $$PCT%"; \
	    echo "=========================================="; \
	done; \
	echo "[*] cuBLAS"; \
	./cuBLAS; \
	echo "vs cuBLAS: 100.0%"; \
	echo "=========================================="

ncurep: all
	@mkdir -p $(PROFILE_DIR)
	ncu --set full --launch-skip 1 --launch-count 1 -o $(PROFILE_DIR)/naive         ./naive
	ncu --set full --launch-skip 1 --launch-count 1 -o $(PROFILE_DIR)/coalesced     ./coalesced
	ncu --set full --launch-skip 1 --launch-count 1 -o $(PROFILE_DIR)/tiled         ./tiled
	ncu --set full --launch-skip 1 --launch-count 1 -o $(PROFILE_DIR)/microtiling   ./microtiling
	ncu --set full --launch-skip 1 --launch-count 1 -o $(PROFILE_DIR)/vectorization ./vectorization
	ncu --set full --launch-skip 1 --launch-count 1 -o $(PROFILE_DIR)/parameterTune ./parameterTune
	ncu --set full --launch-skip 1 --launch-count 1 -o $(PROFILE_DIR)/warptiling    ./warptiling
	ncu --set full --launch-skip 1 --launch-count 1 -o $(PROFILE_DIR)/doublebuffer  ./doublebuffer
	ncu --set full --launch-skip 10 --launch-count 1 -o $(PROFILE_DIR)/cuBLAS        ./cuBLAS

clean:
	rm -f $(TARGETS)
	rm -rf $(PROFILE_DIR)
