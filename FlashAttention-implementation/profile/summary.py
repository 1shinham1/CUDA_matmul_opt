#!/usr/bin/env python3
"""result/*.ncu-rep 에서 핵심 지표만 뽑아 표로 출력한다.

    make summary
    python3 profile/summary.py result/kernel_1.ncu-rep result/kernel_2.ncu-rep

`ncu -i <rep> --csv --page raw` 출력을 파싱한다 (ncu 가 PATH 에 있어야 함).
전체 지표는 `ncu-ui result/kernel_3.ncu-rep` 로 GUI 에서 볼 수 있다.
"""

import csv
import io
import subprocess
import sys
from pathlib import Path

# 텐서코어 파이프 활용도.
#
# ncu 2025.4 (CUDA 13) 에서 cycles_active 지표가 사라졌다. 남은 것은 "발사된
# 명령어" 기준의 hmma_v2 뿐인데, 이 둘은 같은 양을 다른 peak 에 대해 표현한
# 것이라 상수배로 이어진다. HMMA 하나가 텐서 파이프를 2 사이클 점유하므로
# peak(cycles) = 2 x peak(inst) 이고, 따라서 pct 는 정확히 2배가 된다.
#
#   sm_89 / kernel_2 에서 실측 (ncu 12.8 은 두 지표를 모두 내놓는다):
#       cycles_active = 44.381451
#       hmma_v2       = 22.190726     -> 비 2.0000
#
# 그래서 구지표가 있으면 그대로 쓰고, 없으면 신지표 x2 로 메운다.
_TENSOR_CYCLES = "sm__pipe_tensor_op_hmma_cycles_active.sum.pct_of_peak_sustained_active"
_TENSOR_INST = "sm__inst_executed_pipe_tensor_op_hmma_v2.avg.pct_of_peak_sustained_active"


def _tensor_core_pct(d: dict[str, float]) -> float | None:
    v = d.get(_TENSOR_CYCLES)
    if v is not None:
        return v
    v = d.get(_TENSOR_INST)
    return None if v is None else 2.0 * v


# 뱅크 충돌로 낭비된 shared memory wavefront 비율.
# ncu 지표 이름이 아니라 두 지표에서 직접 계산한다.
def _smem_waste_pct(d: dict[str, float]) -> float | None:
    excessive = d.get("derived__memory_l1_wavefronts_shared_excessive")
    total = d.get("l1tex__data_pipe_lsu_wavefronts_mem_shared.sum")
    if excessive is None or not total:
        return None
    return 100.0 * excessive / total


# (표시 라벨, ncu 지표 이름 또는 계산 함수, 배수, 소수점, 설명)
METRICS = [
    ("duration (ms)", "gpu__time_duration.sum", 1.0, 3,
     "커널 실행 시간 (ncu 계측값이라 실측보다 조금 크다)"),
    ("SM thruput %", "sm__throughput.avg.pct_of_peak_sustained_elapsed", 1.0, 1,
     "SM 전체 파이프라인 활용도"),
    ("tensor core %", _tensor_core_pct, 1.0, 1,
     "HMMA(텐서코어) 파이프 활용도"),
    ("DRAM %", "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed", 1.0, 1,
     "메모리 대역폭 활용도 (계산 집약적이라 낮은 게 정상)"),
    ("occupancy %", "sm__warps_active.avg.pct_of_peak_sustained_active", 1.0, 1,
     "achieved occupancy (smem 48KB 라 SM 당 2 CTA 로 묶인다)"),
    ("registers", "launch__registers_per_thread", 1.0, 0,
     "스레드당 레지스터"),
    ("smem conflict (M)", "l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum", 1e-6, 1,
     "shared memory 뱅크 충돌 횟수 (백만) ★ kernel 1 vs 2 를 보라"),
    ("smem waste %", _smem_waste_pct, 1.0, 1,
     "충돌 때문에 낭비된 smem wavefront 비율"),
]


def read_report(path: Path) -> dict[str, float]:
    """ncu 리포트 하나 -> {지표 이름: 값}"""
    out = subprocess.run(
        ["ncu", "-i", str(path), "--csv", "--page", "raw"],
        capture_output=True, text=True, check=True,
    ).stdout

    rows = list(csv.reader(io.StringIO(out)))
    if len(rows) < 3:
        raise RuntimeError(f"{path}: 예상 밖의 CSV 형식")

    header = rows[0]
    # rows[1] 은 단위 행. 실제 데이터는 rows[2] 부터 (커널 1개만 프로파일했다).
    data = rows[2]

    result: dict[str, float] = {}
    for name, value in zip(header, data):
        try:
            v = float(value.replace(",", ""))
        except ValueError:
            continue
        result[name] = v
        # ncu 는 일부 지표에 "TriageXX." 접두어를 붙인다. 접두어 없는 키도 넣어둔다.
        if name.startswith("Triage") and "." in name:
            result.setdefault(name.split(".", 1)[1], v)
    return result


def main(paths: list[str]) -> int:
    reports = []
    for p in paths:
        path = Path(p)
        if not path.exists():
            print(f"[건너뜀] {path} 없음 — 먼저 `make profile` 을 실행하세요")
            continue
        reports.append((path.stem, read_report(path)))

    if not reports:
        print("읽을 리포트가 없습니다. `make profile` 을 먼저 실행하세요.")
        return 1

    label_w = max(len(label) for label, *_ in METRICS) + 2
    col_w = 11

    header = " " * label_w + "".join(f"{n.replace('kernel_', 'k'):>{col_w}}"
                                     for n, _ in reports)
    print()
    print(header)
    print("-" * len(header))

    for label, metric, mult, prec, _ in METRICS:
        cells = []
        for _, data in reports:
            v = metric(data) if callable(metric) else data.get(metric)
            cells.append("-" if v is None else f"{v * mult:,.{prec}f}")
        print(f"{label:<{label_w}}" + "".join(f"{c:>{col_w}}" for c in cells))

    print("-" * len(header))
    print()
    for label, _m, _mult, _p, desc in METRICS:
        print(f"  {label:<18s} {desc}")
    return 0


if __name__ == "__main__":
    args = sys.argv[1:] or sorted(
        str(p) for p in Path("result").glob("kernel_*.ncu-rep"))
    sys.exit(main(args))
