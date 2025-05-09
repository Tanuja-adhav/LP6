#include <iostream>
#include <vector>
#include <ctime>
#include <cuda_runtime.h>

using namespace std;

// CUDA kernel for vector addition (GPU version)
__global__ void vectorAddKernel(float *a, float *b, float *c, int N) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < N) {
        c[index] = a[index] + b[index];
    }
}

int main() {
    int N = 10000000;  // Size of large vectors
    vector<float> a_cpu(N), b_cpu(N), c_cpu(N);  // Vectors on CPU
    vector<float> c_gpu(N);  // Vector to store GPU result

    // Fill vectors with random data
    srand(time(0));
    for (int i = 0; i < N; ++i) {
        a_cpu[i] = static_cast<float>(rand()) / RAND_MAX;
        b_cpu[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // ---------- CPU Vector Addition ----------
    clock_t start_cpu = clock();
    for (int i = 0; i < N; ++i) {
        c_cpu[i] = a_cpu[i] + b_cpu[i];
    }
    clock_t end_cpu = clock();
    double cpu_time = double(end_cpu - start_cpu) / CLOCKS_PER_SEC;
    cout << "CPU Vector Addition Time: " << cpu_time << " seconds" << endl;

    // ---------- GPU Vector Addition ----------
    float *d_a, *d_b, *d_c;

    // Allocate memory on GPU
    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_b, N * sizeof(float));
    cudaMalloc(&d_c, N * sizeof(float));

    // Copy data from CPU to GPU
    cudaMemcpy(d_a, a_cpu.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b_cpu.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    clock_t start_gpu = clock();
    vectorAddKernel<<<numBlocks, blockSize>>>(d_a, d_b, d_c, N);
    cudaDeviceSynchronize();  // Ensure GPU computation is done
    clock_t end_gpu = clock();

    // Copy result back from GPU to CPU
    cudaMemcpy(c_gpu.data(), d_c, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Measure GPU time
    double gpu_time = double(end_gpu - start_gpu) / CLOCKS_PER_SEC;
    cout << "GPU Vector Addition Time: " << gpu_time << " seconds" << endl;

    // Free GPU memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
