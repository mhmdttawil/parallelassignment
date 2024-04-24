#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <openacc.h>

void matrix_multiplication(int *a, int *b, int *c, int n) {
    // Parallelize outer loop with OpenACC directive
    #pragma acc parallel loop collapse(2) copyin(a[0:n*n], b[0:n*n]) copyout(c[0:n*n])
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            int sum = 0;
            for (int k = 0; k < n; k++) {
                sum += a[i * n + k] * b[k * n + j];
            }
            c[i * n + j] = sum;
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

    // Perform matrix multiplication using OpenACC
    matrix_multiplication(a, b, c, n);

    double end = omp_get_wtime();  // Record end time
    printf("Time elapsed for basic OpenACC matrix multiplication: %f seconds\n", end - start);

    free(a);
    free(b);
    free(c);

    return 0;
}
