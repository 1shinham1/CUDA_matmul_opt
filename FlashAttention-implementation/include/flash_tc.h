// [Tensor Core 공용 헤더] 01_flash_tc_naive.cu가 쓰는 WMMA 상수 + 정렬 헬퍼.
// CUDA_matmul_opt의 include/gemm_tc.h와 같은 역할. 커널 자체는 01_flash_tc_naive.cu
// 안에 직접 있다 -- 대신 flash.h의 tiled 커널을 "프로세스 안에서 비교할 빠른
// 기준"으로 재사용한다 (GEMM의 TC 파일들이 cuBLAS를 재사용하는 것과 같은 이유).
#pragma once
#include "flash.h"
#include <mma.h>

using namespace nvcuda;

constexpr int WMMA_M = 16, WMMA_N = 16, WMMA_K = 16;

// 16바이트 경계로 올림 -- WMMA fragment 로드가 정렬된 포인터를 요구하므로,
// shared memory 안에서 서로 다른 타입(half/float) 영역을 이어붙일 때 필요.
__host__ __device__ inline size_t align16(size_t bytes) { return (bytes + 15) & ~size_t(15); }
