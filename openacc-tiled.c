#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <openacc.h>

#define TILE_SIZE 16  // Size of each tile

void tiled_matrix_multiplication(int *a, int *b, int *c, int n) {
    #pragma acc data copyin(a[0:n*n], b[0:n*n]) copyout(c[0:n*n])
    {
        #pragma acc parallel loop collapse(2) gang
        for (int row = 0; row < n; row += TILE_SIZE) {
            for (int col = 0; col < n; col += TILE_SIZE) {
                #pragma acc loop worker
                for (int i = 0; i < TILE_SIZE; i++) {
                    for (int j = 0; j < TILE_SIZE; j++) {
                        int c_row = row + i;
                        int c_col = col + j;

                        if (c_row < n && c_col < n) {
                            int sum = 0;
                            for (int k = 0; k < n; k++) {
                                sum += a[c_row * n + k] * b[k * n + c_col];
                            }
                            c[c_row * n + c_col] = sum;
                        }
                    }
                }
            }
        }
    }
}

int main() {
    int n = 1024;  // Example matrix size
    int *a = (int *)malloc(n * n * sizeof(int));
    int *b = (int *)malloc(n * n * sizeof(int));
    int *c = (int *)malloc(n * n * sizeof(int));

    // Initialize matrices
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            a[i * n + j] = 2;
            b[i * n + j] = 3;
        }
    }

    double start = omp_get_wtime();  // Record start time

    // Perform tiled matrix multiplication using OpenACC
    tiled_matrix_multiplication(a, b, c, n);

    double end = omp_get_wtime();  // Record end time
    printf("Time elapsed for tiled OpenACC matrix multiplication: %f seconds\n", end - start);

    free(a);
    free(b);
    free(c);

    return 0;
}
