#include "cuda_test.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void add(const int *a, int *b) {
    auto index = threadIdx.x;
    b[index] += a[index];
}


CudaAbstractTask cu_add_test(int *a, int *b, int n) {
    cudaStream_t stream;
    if (cudaStreamCreate(&stream) != cudaSuccess) {
        throw CudaException("cudaStreamCreate failed");
    }

    int *da, *db;
    cudaMalloc(&da, n * sizeof(int));
    cudaMalloc(&db, n * sizeof(int));
    cudaMemcpyAsync(da, a, n * sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(db, b, n * sizeof(int), cudaMemcpyHostToDevice, stream);

    add<<<1, n, 0, stream>>>(da, db);

    cudaMemcpyAsync(b, db, sizeof(int) * n, cudaMemcpyDeviceToHost, stream);

    return CudaAbstractTask(stream) << da << db;
}
