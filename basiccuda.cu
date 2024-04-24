#include <stdio.h>

// Define matrix dimensions
#define M 1024 // Number of rows in A and C
#define N 1024 // Number of columns in B and C
#define K 1024 // Number of columns in A and rows in B

// CUDA kernel to compute matrix multiplication
__global__ void matrixMultiply(float *A, float *B, float *C) {
    // Determine row and column index for this thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Only compute if within bounds
    if (row < M && col < N) {
        float sum = 0.0;
        for (int i = 0; i < K; i++) {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

int main() {
    // Allocate and initialize host matrices
    float *h_A, *h_B, *h_C;
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    h_A = (float *)malloc(size_A);
    h_B = (float *)malloc(size_B);
    h_C = (float *)malloc(size_C);

    // Initialize A and B with sample data
    for (int i = 0; i < M * K; i++) {
        h_A[i] = static_cast<float>(i);
    }
    for (int i = 0; i < K * N; i++) {
        h_B[i] = static_cast<float>(i);
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    // Define CUDA grid and block dimensions
    dim3 blockDim(16, 16); // Each block has 16x16 threads
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);

    // Launch CUDA kernel
    matrixMultiply<<<gridDim, blockDim>>>(d_A, d_B, d_C);

    // Copy results from device to host
    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
