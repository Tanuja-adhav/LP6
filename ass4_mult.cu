#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <omp.h>
#include <cuda_runtime.h>
#include <chrono>

using namespace std;

// Function to multiply matrices sequentially on the CPU
void matrixMultiplyCPU(const vector<vector<float>>& A, const vector<vector<float>>& B, vector<vector<float>>& C, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            C[i][j] = 0;
            for (int k = 0; k < N; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

// CUDA kernel for matrix multiplication
__global__ void matrixMultiplyGPU(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float value = 0;
        for (int k = 0; k < N; ++k) {
            value += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = value;
    }
}

int main() {
    int N = 1024;  // Matrix size (NxN)

    // Create random matrices on the CPU
    vector<vector<float>> A(N, vector<float>(N)), B(N, vector<float>(N)), C(N, vector<float>(N));

    srand(time(0));

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i][j] = rand() % 100;
            B[i][j] = rand() % 100;
        }
    }

    // ---------- CPU Matrix Multiplication ----------
    auto start_cpu = chrono::high_resolution_clock::now();
    matrixMultiplyCPU(A, B, C, N);
    auto end_cpu = chrono::high_resolution_clock::now();
    chrono::duration<float> duration_cpu = end_cpu - start_cpu;
    cout << "CPU Matrix Multiplication Time: " << duration_cpu.count() << " seconds\n";

    // ---------- GPU Matrix Multiplication ----------
    float *d_A, *d_B, *d_C;
    size_t size = N * N * sizeof(float);

    // Allocate memory on GPU
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // Copy matrices A and B from host to device
    cudaMemcpy(d_A, A[0].data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B[0].data(), size, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(N / 16, N / 16);

    auto start_gpu = chrono::high_resolution_clock::now();
    matrixMultiplyGPU<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    auto end_gpu = chrono::high_resolution_clock::now();
    chrono::duration<float> duration_gpu = end_gpu - start_gpu;

    cout << "GPU Matrix Multiplication Time: " << duration_gpu.count() << " seconds\n";

    // Free GPU memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}



**********************************************************************
** Visual Studio 2022 Developer Command Prompt v17.13.5
** Copyright (c) 2022 Microsoft Corporation
**********************************************************************
[vcvarsall.bat] Environment initialized for: 'x64'

C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools>nvcc -o nspdp_gpu.exe nspdp_gpu.cu
nvcc warning : Support for offline compilation for architectures prior to '<compute/sm/lto>_75' will be removed in a future release (Use -Wno-deprecated-gpu-targets to suppress warning).
nspdp_gpu.cu
c1xx: fatal error C1083: Cannot open source file: 'nspdp_gpu.cu': No such file or directory

C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools>cd C:\Users\admin\OneDrive\Documents\BE_LP5

C:\Users\admin\OneDrive\Documents\BE_LP5>nvcc -o miniProject2.exe miniProject2.cu
nvcc warning : Support for offline compilation for architectures prior to '<compute/sm/lto>_75' will be removed in a future release (Use -Wno-deprecated-gpu-targets to suppress warning).
miniProject2.cu
tmpxft_00005bf8_00000000-7_miniProject2.cudafe1.cpp
   Creating library miniProject2.lib and object miniProject2.exp

C:\Users\admin\OneDrive\Documents\BE_LP5>miniProject2.exe
dp[0] = 1
dp[1] = 1
dp[2] = 1
dp[3] = 1
dp[4] = 1
dp[5] = 3
dp[6] = 5
dp[7] = 9
dp[8] = 15
dp[9] = 4522090
dp[10] = 22479271
dp[11] = 49677121
dp[12] = 70583309
dp[13] = 100468080
dp[14] = 160762149
dp[15] = 214764625
dp[16] = 235015377
dp[17] = 271519382
dp[18] = 329323153
dp[19] = 350688401
