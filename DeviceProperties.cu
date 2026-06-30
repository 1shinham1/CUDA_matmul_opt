#include <stdio.h>
#include <stdlib.h>

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    printf("SM당 최대 SMEM:                     %zu KB\n", prop.sharedMemPerBlock / 1024);
    printf("블록당 최대 스레드:                     %d\n", prop.maxThreadsPerBlock);
    printf("SM당 최대 레지스터:                     %d\n", prop.regsPerBlock);
    printf("Warp 크기:                      %d\n", prop.warpSize);
    printf("SM 수:                          %d\n", prop.multiProcessorCount);

    printf("SM당 최대 스레드:                       %d\n", prop.maxThreadsPerMultiProcessor);
    printf("SM당 최대 SMEM:                     %zu B\n", prop.sharedMemPerMultiprocessor);
    printf("총 글로벌 메모리:                       %zu MB\n", prop.totalGlobalMem / 1024 / 1024);
    printf("L2 캐시 크기:                       %d KB\n", prop.l2CacheSize / 1024);
    printf("메모리 버스 폭:                     %d bit\n", prop.memoryBusWidth);
    printf("Compute Capability:                 %d.%d\n", prop.major, prop.minor);
}
/*
SM당 최대 SMEM:       48 KB
블록당 최대 스레드:    1024 --=----> warp이 32개
SM당 최대 레지스터:     65536 ------> /1024 = 64개/register 
Warp 크기:            32
SM 수:                128
*/

/*SM당 최대 SMEM에 따른 BK, BM, BN 이론값 구하기
As + Bs ≤ 48KB
As = BM x BK x 4byte
Bs = BK x BN x 4byte
(BM x BK + BK x BN) x 4 ≤ 48 x 1024
BK x (BM + BN) ≤ 12288

따라서
BK=16, BM=BN=256: 16 x 512 = 8192  ✓ (32KB 사용, 가능)
그러나 딱 48을 맞추면
BK=48, BM=BN=128: 48 x 256 = 12288 딱 48KB
*/

/*SM당 최대 레지스터에 따른 thread당 register = 64 
따락서 
TR x TC = 64로 TR=TC=8일때 최대
*/


/*최종 결론
BK = 48, BM = BN = 128
SMEM = 48KB
스레드 256개 = warp 8개
TR=TC=8, 레지스터 64개 ------->로 하려했는데 NUM_THREADS / (BK/4)값이 정수로 안나와서 As 로드(float4로 읽고 전치해서 저장할)때 문제 발생


BK = 64, BM = BN = 96
SMEM = 48KB
스레드 (BM/TR) x (BN/TC)개 = 144개 warp 4.5개 --------->로 하려했는데 threa수가 32배수가 아니여서 문제
TR=TC=8, 레지스터 64개

3가지 조건
1. BK가 4의 배수                (float4 로드)
2. NUM_THREADS가 BK/4의 배수    (루프 나누어떨어짐)
3. NUM_THREADS가 32의 배수      (warp 단위 맞춤)

BM=BN=128, BK=32
SMEM = 32KB
warp = 8개  (최종 선택)
*/