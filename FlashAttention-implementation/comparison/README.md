# comparison — 현재 baseline vs. official FA1, head-to-head

[`README.md`](../README.md)가 이 프로젝트의 현재 baseline(01_flash_tc_naive)을
설명한다면, 이 디렉토리는 그 baseline을
[`official/`](../official/) — 논문의 실제 CUTLASS 기반 FlashAttention-1
커널 — 과 나란히 놓고 얼마나 격차가 있는지 보여준다. forward pass, fp16
기준.

## 구성

- `compare.py` — 고정 그리드(batch=4, heads=8, head_dim=64) 위에서 01의
  `results/results_01_tc_naive_noncausal.csv`를 읽어 official과 병합하고,
  `01_tc_naive_vs_official_x` 비율을 계산해 `results/compare.csv`로 저장한다.

## How to run

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate cuda_env   # flash_attn(official) 설치돼있어야 함
cd FlashAttention-implementation
make run                       # results_01_tc_naive_*.csv 생성 (없으면 먼저 실행)
cd comparison
python compare.py              # 01 vs official, 고정 그리드 -> ../results/compare.csv
```

official 쪽만 스크립트 안에서 새로 측정하고(`flash_attn_unpadded_qkvpacked_func`),
자체 구현 쪽은 이미 생성된 CSV를 그대로 재사용한다.
