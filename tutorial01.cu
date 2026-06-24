#include <stdio.h>
#include <stdlib.h>

#define N 10000000

__global__ void vector_add(float *out, float *a, float *b, int n) {
    for(int i = 0; i < n; i++){
        out[i] = a[i] + b[i];
    }
}

int main(){
    float *a, *b, *out; 
    float *d_a, *d_b, *d_out;

    //메모리할당
    a   = (float*)malloc(sizeof(float) * N);
    b   = (float*)malloc(sizeof(float) * N);
    out = (float*)malloc(sizeof(float) * N);

    //array 초기ㅇ화
    for(int i = 0; i < N; i++){
        a[i] = 1.0f;
        b[i] = 2.0f;
    }

    // device memory(GPU)할당하ㅣㄹ
    cudaMalloc((void**)&d_a, sizeof(float) * N);
    cudaMalloc((void**)&d_b, sizeof(float) * N);
    cudaMalloc((void**)&d_out, sizeof(float) * N);

    // host(CPU)에서 device memory(GPU)로 옮기기
    cudaMemcpy(d_a, a, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(float) * N, cudaMemcpyHostToDevice);

    // Main function
    vector_add<<<1,1>>>(d_out, d_a, d_b, N);

    cudaDeviceSynchronize(); // GPU가 실제로 끝나기 전에 프로그램이 종료되는거여서 검증 시간이 이상해서 추가
    cudaMemcpy(out, d_out, sizeof(float)*N, cudaMemcpyDeviceToHost);
    printf("out[0] = %.1f\n", out[0]);
    printf("out[N-1] = %.1f\n", out[N-1]);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);
    free(a);
    free(b);
    free(out);
    return 0;
}

//nvcc 파일.cu
//time : 전체 실행 시간
//nvprof : 더 상세하게 알려줌
//ncu : kernel 상세분석 roofline 그래프 그려줌
//nsys : 전체 타임라인 분석(타임라인 그래프)