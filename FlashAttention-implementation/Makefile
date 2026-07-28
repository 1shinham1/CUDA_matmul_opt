NVCC ?= nvcc
ARCH ?= sm_89

ROOT_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
BIN_DIR := $(ROOT_DIR)bin
INCLUDE_DIR := $(ROOT_DIR)include
SRC_DIR := $(ROOT_DIR)src
PROFILE_DIR := $(ROOT_DIR)results/profiles

NVCCFLAGS := -O3 -std=c++17 -arch=$(ARCH) -I$(INCLUDE_DIR) -Xcompiler -fopenmp

TARGETS := \
	$(BIN_DIR)/00_attention_naive \
	$(BIN_DIR)/01_flash_tc_naive \
	$(BIN_DIR)/02_flash_tc_async

.PHONY: all run profile clean

all: $(BIN_DIR) $(TARGETS)

$(BIN_DIR):
	mkdir -p $@

$(BIN_DIR)/00_attention_naive: $(SRC_DIR)/00_attention_naive.cu $(INCLUDE_DIR)/flash.h | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

$(BIN_DIR)/01_flash_tc_naive: $(SRC_DIR)/01_flash_tc_naive.cu $(INCLUDE_DIR)/flash.h $(INCLUDE_DIR)/flash_tc.h | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

$(BIN_DIR)/02_flash_tc_async: $(SRC_DIR)/02_flash_tc_async.cu $(INCLUDE_DIR)/flash.h $(INCLUDE_DIR)/flash_tc.h | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

# ─── 전체 벤치마크 실행 ──────────────────────────────────────
run: all
	@bash $(ROOT_DIR)scripts/benchmark.sh --skip-build

# ─── NCU 프로파일링 (바이너리가 새로 빌드된 것만 재실행) ────────────
PROFILES := \
	$(PROFILE_DIR)/00_attention_naive.ncu-rep \
	$(PROFILE_DIR)/01_tc_naive.ncu-rep \
	$(PROFILE_DIR)/02_tc_async.ncu-rep

profile: $(PROFILES)

$(PROFILE_DIR):
	mkdir -p $(PROFILE_DIR)

$(PROFILE_DIR)/00_attention_naive.ncu-rep: $(BIN_DIR)/00_attention_naive | $(PROFILE_DIR)
	ncu -f --set full --launch-skip 5 --launch-count 1 -o $(PROFILE_DIR)/00_attention_naive $(BIN_DIR)/00_attention_naive

$(PROFILE_DIR)/01_tc_naive.ncu-rep: $(BIN_DIR)/01_flash_tc_naive | $(PROFILE_DIR)
	ncu -f --set full --launch-skip 5 --launch-count 1 -o $(PROFILE_DIR)/01_tc_naive $(BIN_DIR)/01_flash_tc_naive

$(PROFILE_DIR)/02_tc_async.ncu-rep: $(BIN_DIR)/02_flash_tc_async | $(PROFILE_DIR)
	ncu -f --set full --launch-skip 5 --launch-count 1 -o $(PROFILE_DIR)/02_tc_async $(BIN_DIR)/02_flash_tc_async

clean:
	rm -rf $(BIN_DIR)
	rm -rf $(PROFILE_DIR)
