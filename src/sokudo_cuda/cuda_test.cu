#include "cuda_test.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void add(const int *a, int *b) {
    auto index = threadIdx.x;
    b[index] += a[index];
}


CudaAbstractTask cu_add_test(int *a, int *b, int n) {
    CudaError err;
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    int *da, *db;
    err << cudaMalloc(&da, n * sizeof(int));
    err << cudaMalloc(&db, n * sizeof(int));
    err << cudaMemcpyAsync(da, a, n * sizeof(int), cudaMemcpyHostToDevice, stream);
    err << cudaMemcpyAsync(db, b, n * sizeof(int), cudaMemcpyHostToDevice, stream);

    add<<<1, n, 0, stream>>>(da, db);

    err << cudaMemcpyAsync(b, db, sizeof(int) * n, cudaMemcpyDeviceToHost, stream);

    return CudaAbstractTask(stream) << da << db;
}
