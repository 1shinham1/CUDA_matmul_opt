# CUDA Matrix Multiplication Optimization

NVIDIA GPU(sm_89)에서 4096×4096 float32 행렬 곱셈(SGEMM)을 Naive 구현부터 시작해  
단계적으로 최적화하여 cuBLAS 성능의 **90.9%** 까지 도달한 프로젝트.

---

## Results

| Step | Kernel | Time (ms) | TFLOPS | vs cuBLAS |
|------|--------|-----------|--------|-----------|
| 1 | Naive | 199.536 | 0.69 | 1.1% |
| 2 | Coalesced | 26.849 | 5.12 | 8.1% |
| 3 | Tiling | 21.997 | 6.25 | 9.9% |
| 4 | Microtiling | 3.361 | 40.90 | 65.1% |
| 5 | Vectorization | 2.984 | 46.06 | 73.3% |
| 6 | Parameter Tuning | 2.800 | 49.09 | 78.1% |
| 7 | Warptiling | 2.584 | 53.18 | 84.6% |
| 8 | Double Buffering | 2.407 | 57.10 | 90.9% |
| * | cuBLAS (baseline) | 2.187 | 62.84 | 100.0% |

---

## Optimization Breakdown
단계별 병목 분석 및 해결
### Step 1 — Naive : Memory-Bound
각 스레드가 Global Memory에서 A·B 원소를 직접 읽는 기본 구현. Non-coalesced 접근으로 워프 내 32개 스레드가 같은 데이터를 32번 따로 로드한다. ncu 기준 Compute Throughput 12%, Memory(L1/TEX) Throughput 99% — L1 캐시가 포화 상태가 되는 전형적인 Memory-Bound.

### Step 2 — Coalesced : Memory-Bound
워프 내 스레드가 인접 주소를 읽도록 인덱싱을 재배열하여 동일 캐시 라인을 한 번의 트랜잭션으로 처리. L1 캐시 요청 수가 줄며 Compute Throughput이 95%까지 상승(×7.4 speedup). 다만 같은 데이터를 반복 로드하는 구조적 문제는 남아 있어 Shared Memory를 통한 재사용이 필요했다.

### Step 3 — Tiling (Shared Memory) : Compute-Bound (저밀도)
TILE×TILE 블록을 Shared Memory에 적재해 블록 내 스레드들이 재사용. 단, 스레드 1개 = 출력 원소 1개 구조라 SMEM 1회 읽기당 FMA 연산이 거의 1:1 비율. ncu roofline상 이미 compute-bound 영역에 도달했지만(Compute 80% / Memory 80%, "Balanced Throughput" 경고), 연산 밀도가 낮아 FP32 peak의 **7%**에 그쳤다.

### Step 4 — Microtiling : Compute-Bound ⭐
스레드 1개가 TR×TC개의 출력 원소를 레지스터에 누적하도록 변경. SMEM 1회 접근당 FMA 연산 수(Arithmetic Intensity)가 급증하며, 동일한 compute-bound 상태 안에서 연산 명령 밀도를 끌어올렸다. 21.997 ms → 3.361 ms (×6.5 단축), FP32 peak 대비 48%, cuBLAS 대비 65.1%. 다만 여전히 float1(4B) 단위로 로드하여 GPU 캐시 라인(128B)을 온전히 활용하지 못했다.

### Step 5 — Vectorization : Compute-Bound
float4를 사용해 GMEM에서 128B(16B, LDG.128) 단위로 로드하고, SMEM에 저장할 때 전치하여 이후 계산 루프에서 연속 접근이 가능하도록 했다. 메모리 명령 수가 1/4로 줄고 메모리 버스 활용률이 상승. 2.984 ms / 46.06 TFLOPS. 단, BM=BN=64, BK=16 설정으로 SMEM을 8KB(48KB 한도 중)만 사용하고 있어 추가 튜닝 여지가 있었다.

### Step 6 — Parameter Tuning : Compute-Bound
BM=128, BN=128, BK=32, TM=TN=8로 RTX 4090의 SMEM 한도(48KB)와 레지스터 파일 크기에 맞게 재탐색. 다만 64개 스레드의 threadRow/threadCol이 제각각 분산되어 있어, 워프 내 32개 스레드가 SMEM의 같은 뱅크를 동시에 요청하면서 Bank Conflict가 발생하기 쉬운 구조였다. → 워프 단위 접근 정렬이 다음 과제로 남음.

### Step 7 — Warptiling : Compute-Bound
Thread Block → Warp → Thread의 3단 계층적 타일링을 도입. 워프 내 스레드들이 출력 타일을 협력 분담하면서 SMEM 접근 패턴을 워프 단위로 정렬, Bank Conflict를 줄이고 레지스터 재사용률을 높였다. 2.584 ms / 53.18 TFLOPS. 이 단계에서 Load와 Compute가 순차적으로 실행되는 점이 다음 개선 대상으로 식별됨.

### Step 8 — Double Buffering : Compute-Bound
cp.async 비동기 복사로 SMEM을 2개로 나누어, 한쪽에서 다음 타일을 로드하는 동안 다른 쪽에서 현재 타일을 연산하는 파이프라인 구성. 메모리 레이턴시를 연산 뒤로 숨겨(overlap) 2.407 ms / 57.10 TFLOPS 달성, cuBLAS 대비 90.9%.

핵심 인사이트
* Naive → Coalesced: 문제는 DRAM 대역폭이 아니라 L1/TEX 캐시 트랜잭션 효율이었다(DRAM Throughput은 두 단계 모두 1% 미만).
* Tiling → Microtiling: ncu Compute Throughput은 오히려 80% → 62%로 줄었지만, FP32 peak 대비 처리량은 7% → 48%로 급증. SASS instruction mix상 LDS:FFMA 비율이 약 1:1(Tiling)에서 FFMA가 압도적으로 많은 비율(Microtiling)로 바뀐 것이 근거.
* Compute-Bound의 두 얼굴: roofline상 "compute-bound"는 메모리 대역폭이 한계가 아니라는 뜻일 뿐, 연산이 효율적이라는 보장은 아니다. Tiling은 이미 compute-bound였지만 연산 밀도가 낮아 비효율적으로 바빴던 경우.

--

## Project Structure

```
matmul_project/
├── tutorial/           # CUDA 병렬 프로그래밍의 기본 패턴을 익히는 예제 (커널 함수, 스레드 인덱싱, Host ↔ Device 메모리 관리)
├── naive.cu
├── coalesced.cu
├── tiled.cu
├── microtiling.cu
├── vectorization.cu
├── parameterTune.cu
├── warptiling.cu
├── doublebuffer.cu
├── cuBLAS.cu
├── DeviceProperties.cu
├── Makefile
└── profiles/           # ncu-rep 프로파일 결과 (make profile 후 생성)
```

---

## Usage

### Build

```bash
make          # 전체 빌드
make naive    # 특정 타겟만 빌드
```

### Run

```bash
make run      # 전체 빌드 후 Step 1~8 + cuBLAS 순서로 실행, cuBLAS 대비 % 출력
```

### Profile (Nsight Compute)

```bash
make profile  # 전체 빌드 후 각 커널 프로파일링 → profiles/ 저장
```

- 워밍업 실행(1회)을 건너뛰고 두 번째 실행만 캡처 (`--launch-skip 1 --launch-count 1`)
- 결과는 `profiles/*.ncu-rep`로 저장, Nsight Compute GUI에서 열람 가능

```bash
ncu-ui profiles/doublebuffer.ncu-rep
```

### Clean

```bash
make clean    # 바이너리 + profiles/ 삭제
```

---

## Requirements

| 항목 | 버전 |
|------|------|
| GPU Architecture | sm_89 (RTX 4000 series / Ada Lovelace) |
| CUDA Toolkit | 12.x 이상 권장 |
| nvcc | CUDA Toolkit 포함 |
| Nsight Compute | `make profile` 사용 시 필요 |
