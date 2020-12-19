#include "cuda_test.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void add(int *a, int *b) {
    int index = threadIdx.x;
    b[index] += a[index];
}


void cu_add_test(int *a, int *b, int n) {
    int *da, *db;
    cudaMalloc(&da, n * sizeof(int));
    cudaMalloc(&db, n * sizeof(int));
    cudaMemcpy(da, a, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(db, b, n * sizeof(int), cudaMemcpyHostToDevice);

    add<<<1, n>>>(da, db);
    cudaDeviceSynchronize();

    cudaMemcpy(b, db, sizeof(int) * n, cudaMemcpyDeviceToHost);
}
