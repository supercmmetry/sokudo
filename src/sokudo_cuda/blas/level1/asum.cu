#include <cublas.h>
#include "asum.h"

CudaAbstractTask cu_sasum(float *a, float *res, uint64_t n) {
    CudaError err;
    CublasError berr;
    cublasHandle_t handle;
    berr << cublasCreate_v2(&handle);

    cudaStream_t stream;
    err << cudaStreamCreate(&stream);

    berr << cublasSetStream_v2(handle, stream);

    float *dx;
    err << cudaMalloc(&dx, n * sizeof(float));
    err << cudaMemcpyAsync(dx, a, n * sizeof(float), cudaMemcpyHostToDevice, stream);

    berr << cublasSasum_v2(handle, n, dx, 1, res);
    return CudaAbstractTask(stream, handle) << dx;
}
