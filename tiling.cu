#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define BLOCK_SIZE 16  // Size of each tile

__global__ void tiled_matrix_mult(int *a, int *b, int *c, int n) {
    __shared__ int shared_a[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int shared_b[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int sum = 0;

    for (int tile = 0; tile < (n + BLOCK_SIZE - 1) / BLOCK_SIZE; tile++) {
        // Load data into shared memory
        if (row < n && tile * BLOCK_SIZE + threadIdx.x < n) {
            shared_a[threadIdx.y][threadIdx.x] = a[row * n + tile * BLOCK_SIZE + threadIdx.x];
        } else {
            shared_a[threadIdx.y][threadIdx.x] = 0;
        }

        if (col < n && tile * BLOCK_SIZE + threadIdx.y < n) {
            shared_b[threadIdx.y][threadIdx.x] = b[(tile * BLOCK_SIZE + threadIdx.y) * n + col];
        } else {
            shared_b[threadIdx.y][threadIdx.x] = 0;
        }

        __syncthreads();  // Ensure all threads in the block have loaded the data

        // Perform the dot product within the tile
        for (int k = 0; k < BLOCK_SIZE; k++) {
            sum += shared_a[threadIdx.y][k] * shared_b[k][threadIdx.x];
        }

        __syncthreads();  // Ensure all computations in the tile are done before moving to the next tile
    }

    if (row < n && col < n) {
        c[row * n + col] = sum;
    }
}

int main(int argc, char const *argv[]) {
    int n;

    for (n = 64; n <= 8192; n *= 2) {
        int *h_a, *h_b, *h_c;
        cudaMallocHost((void **) &h_a, sizeof(int) * n * n);
        cudaMallocHost((void **) &h_b, sizeof(int) * n * n);
        cudaMallocHost((void **) &h_c, sizeof(int) * n * n);

        // Initialize matrices A and B
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                h_a[i * n + j] = 2;
                h_b[i * n + j] = 3;
            }
        }

        float tiled_gpu_elapsed_time_ms;

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        int *d_a, *d_b, *d_c;
        cudaMalloc((void **) &d_a, sizeof(int) * n * n);
        cudaMalloc((void **) &d_b, sizeof(int) * n * n);
        cudaMalloc((void **) &d_c, sizeof(int) * n * n);

        cudaMemcpy(d_a, h_a, sizeof(int) * n * n, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b, sizeof(int) * n * n, cudaMemcpyHostToDevice);

        unsigned int grid_rows = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        unsigned int grid_cols = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        dim3 dimGrid(grid_cols, grid_rows);
        dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

        cudaEventRecord(start, 0);
        tiled_matrix_mult<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, n);

        cudaDeviceSynchronize();  // Ensure the kernel completes before timing

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        cudaEventElapsedTime(&tiled_gpu_elapsed_time_ms, start, stop);

        cudaMemcpy(h_c, d_c, sizeof(int) * n * n, cudaMemcpyDeviceToHost);

        printf("Time elapsed on tiled GPU matrix multiplication of %dx%d: %f ms.\n", n, n, tiled_gpu_elapsed_time_ms);

        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        cudaFreeHost(h_a);
        cudaFreeHost(h_b);
        cudaFreeHost(h_c);
    }

    return 0;
}
