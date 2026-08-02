# ═════════════════════════════════════════════════════════════════════════════
#  my_kernel — 커널 1~3 독립 빌드 + ncu 프로파일링
#
#    make              커널별 실행 파일 -> bin/kernel_N
#    make run          bin/kernel_N 전부 실행 (시간/TFLOPs 출력)
#    make profile      ncu --set full 로 result/kernel_N.ncu-rep 생성
#    make summary      생성된 리포트에서 핵심 지표만 표로 출력
#    make clean        bin/ result/ 삭제
#
#  파라미터 (전부 덮어쓰기 가능):
#    make profile SEQ_LEN=8192 BATCH=2 HEADS=16 WARMUP=20
#    make profile KERNELS="1 2"          # 일부만
#    make ARCH=80                        # 타깃 아키텍처 고정
# ═════════════════════════════════════════════════════════════════════════════

KERNELS ?= 1 2 3

# 워크로드 (bench.py 의 seq_len=4096 케이스와 동일)
SEQ_LEN ?= 4096
BATCH   ?= 4
HEADS   ?= 16
WARMUP  ?= 10

NVCC ?= nvcc
NCU  ?= ncu

# 현재 GPU 의 compute capability 를 자동 감지 (예: 8.9 -> 89)
ARCH ?= $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
                | head -1 | tr -d '. ' )
ifeq ($(strip $(ARCH)),)
  ARCH := 80
endif

NVCCFLAGS := -std=c++20 -O3 \
             --use_fast_math \
             --expt-relaxed-constexpr \
             -Xcudafe="--diag_suppress=3189" \
             -gencode arch=compute_$(ARCH),code=sm_$(ARCH) \
             --generate-line-info \
             --resource-usage \
             -Xptxas=-warn-spills \
             -Xptxas=-warn-lmem-usage \
             -U__CUDA_NO_HALF_OPERATORS__ \
             -U__CUDA_NO_HALF_CONVERSIONS__ \
             -U__CUDA_NO_HALF2_OPERATORS__ \
             -U__CUDA_NO_BFLOAT16_CONVERSIONS__ \
             -I. -Iinclude -Isrc

# --set full            : 모든 섹션 수집 (커널 replay 여러 번)
# --profile-from-start off : harness 의 cudaProfilerStart() 부터 수집
# --launch-count 1      : 그 이후 딱 1개 launch 만
# --import-source yes   : 소스를 리포트에 임베드 (-lineinfo 와 함께)
NCUFLAGS := --set full \
            --profile-from-start off \
            --launch-count 1 \
            --import-source yes \
            --force-overwrite

BINS := $(addprefix bin/kernel_,$(KERNELS))
REPS := $(addsuffix .ncu-rep,$(addprefix result/kernel_,$(KERNELS)))

HEADERS := $(wildcard include/*.h include/*.cuh)

# --resource-usage 출력이 커널당 7줄이라 경계가 안 보인다. 구분선을 끼워 넣는다.
RULE := ─────────────────────────────────────────────────────────

.PHONY: all run profile summary clean help
.DEFAULT_GOAL := all

all: $(BINS)

bin result:
	@mkdir -p $@

# 커널 하나 = 실행 파일 하나. harness.cu 가 src/kernel_N.cu 를 #include 한다.
# (파일명만 넘기고 실제 탐색은 NVCCFLAGS 의 -Isrc 가 처리한다)
bin/kernel_%: profile/harness.cu src/kernel_%.cu $(HEADERS) | bin
	@echo ""
	@echo "── kernel_$*  (sm_$(ARCH)) $(RULE)"
	@$(NVCC) $(NVCCFLAGS) \
	    -DMK_KERNEL_SRC='"kernel_$*.cu"' -DMK_NS=kernel_$* \
	    profile/harness.cu -o $@

run: $(BINS)
	@echo "seq_len=$(SEQ_LEN) batch=$(BATCH) heads=$(HEADS) warmup=$(WARMUP)"
	@for b in $(BINS); do ./$$b $(SEQ_LEN) $(BATCH) $(HEADS) $(WARMUP); done

profile: $(REPS)
	@echo
	@echo "리포트: $(REPS)"
	@echo "  make summary        핵심 지표 표로 보기"
	@echo "  ncu-ui result/kernel_3.ncu-rep    GUI 로 열기"

result/kernel_%.ncu-rep: bin/kernel_% | result
	@echo "[ncu ] $@"
	@$(NCU) $(NCUFLAGS) -o result/kernel_$* \
	    ./bin/kernel_$* $(SEQ_LEN) $(BATCH) $(HEADS) $(WARMUP) > /dev/null

summary:
	@python3 profile/summary.py $(REPS)

clean:
	rm -rf bin result

help:
	@sed -n '2,13p' Makefile | sed 's|^#\ \?||'
	@echo
	@echo "현재 설정: ARCH=sm_$(ARCH)  SEQ_LEN=$(SEQ_LEN) BATCH=$(BATCH) HEADS=$(HEADS) WARMUP=$(WARMUP)"
