#include <cublas.h>
#include "asum.h"

CudaAbstractTask cu_sasum(float *a, float *res, uint64_t n, uint64_t incx) {
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
    auto p = n / incx + (n % incx != 0);
    berr << cublasSasum_v2(handle, p, dx, incx, res);
    return CudaAbstractTask(stream, handle) << dx;
}

CudaAbstractTask cu_dasum(double *a, double *res, uint64_t n, uint64_t incx) {
    CudaError err;
    CublasError berr;
    cublasHandle_t handle;
    berr << cublasCreate_v2(&handle);

    cudaStream_t stream;
    err << cudaStreamCreate(&stream);

    berr << cublasSetStream_v2(handle, stream);

    double *dx;
    err << cudaMalloc(&dx, n * sizeof(double));
    err << cudaMemcpyAsync(dx, a, n * sizeof(double), cudaMemcpyHostToDevice, stream);
    auto p = n / incx + (n % incx != 0);
    berr << cublasDasum_v2(handle, p, dx, incx, res);
    return CudaAbstractTask(stream, handle) << dx;
}

CudaAbstractTask cu_scasum(float2 *a, float *res, uint64_t n, uint64_t incx) {
    CudaError err;
    CublasError berr;
    cublasHandle_t handle;
    berr << cublasCreate_v2(&handle);

    cudaStream_t stream;
    err << cudaStreamCreate(&stream);

    berr << cublasSetStream_v2(handle, stream);

    double *dx;
    err << cudaMalloc(&dx, n * sizeof(double));
    err << cudaMemcpyAsync(dx, a, n * sizeof(double), cudaMemcpyHostToDevice, stream);
    auto p = n / incx + (n % incx != 0);
    berr << cublasScasum_v2(handle, p, reinterpret_cast<const cuComplex *>(dx), incx, res);
    return CudaAbstractTask(stream, handle) << dx;
}
