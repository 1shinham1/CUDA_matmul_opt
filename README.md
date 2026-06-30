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

### Step 1 — Naive
- 각 스레드가 Global Memory에서 A·B 원소를 직접 읽는 기본 구현
- Non-coalesced 접근 + 데이터 재사용 없음 → **Memory-Bound**

### Step 2 — Coalesced Memory Access
- 인접 스레드가 인접 주소를 읽도록 스레드 인덱싱 재배열
- DRAM 트랜잭션 수 감소 → **×7.4 speedup**

### Step 3 — Shared Memory Tiling
- TILE=32 단위로 Shared Memory에 데이터 적재 후 재사용
- Global Memory 접근 횟수 N/TILE 배 감소 → **×9.1 speedup**

### Step 4 — Microtiling ⭐
- 스레드 1개가 BN×BM 서브타일을 담당, 레지스터에 결과 누적
- Arithmetic Intensity 급상승 → **Memory-Bound → Compute-Bound 전환**
- 단일 단계 최대 개선: 21.997 ms → 3.361 ms (**×6.5 추가 단축**)

### Step 5 — Vectorization
- `float4` 벡터 타입으로 128-bit 로드/스토어(LDS.128) 적용
- 메모리 명령 수 1/4 감소 → **×66.9 speedup vs Naive**

### Step 6 — Parameter Tuning
- BM, BN, BK, TM, TN을 sm_89 Shared Memory(48 KB) · 레지스터 파일에 맞게 탐색
- Occupancy 최대화 → 메모리 레이턴시를 연산으로 은닉(Latency Hiding)

### Step 7 — Warptiling
- 블록 → 워프 → 스레드 3단 계층적 타일링
- 워프 단위로 Shared Memory 접근 패턴 정렬 → 레지스터 재사용률 향상

### Step 8 — Double Buffering
- `cp.async` 비동기 복사로 현재 타일 연산과 다음 타일 로드를 overlap
- 메모리 레이턴시를 파이프라인 뒤에 숨김 → **cuBLAS 대비 90.9%**

---

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
