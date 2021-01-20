#include <cublas.h>
#include "asum.h"

CudaAbstractTask cu_sasum(float *a, float *res, uint64_t n, uint64_t incx, uint64_t size) {
    CudaError err;
    CublasError berr;
    cublasHandle_t handle;
    berr << cublasCreate_v2(&handle);

    cudaStream_t stream;
    err << cudaStreamCreate(&stream);

    berr << cublasSetStream_v2(handle, stream);

    float *dx;
    err << cudaMalloc(&dx, size * sizeof(float));
    err << cudaMemcpyAsync(dx, a, size * sizeof(float), cudaMemcpyHostToDevice, stream);
    berr << cublasSasum_v2(handle, n, dx, incx, res);
    return CudaAbstractTask(stream, handle) << dx;
}

CudaAbstractTask cu_dasum(double *a, double *res, uint64_t n, uint64_t incx, uint64_t size) {
    CudaError err;
    CublasError berr;
    cublasHandle_t handle;
    berr << cublasCreate_v2(&handle);

    cudaStream_t stream;
    err << cudaStreamCreate(&stream);

    berr << cublasSetStream_v2(handle, stream);

    double *dx;
    err << cudaMalloc(&dx, size * sizeof(double));
    err << cudaMemcpyAsync(dx, a, size * sizeof(double), cudaMemcpyHostToDevice, stream);
    berr << cublasDasum_v2(handle, n, dx, incx, res);
    return CudaAbstractTask(stream, handle) << dx;
}

CudaAbstractTask cu_scasum(void *a, float *res, uint64_t n, uint64_t incx, uint64_t size) {
    CudaError err;
    CublasError berr;
    cublasHandle_t handle;
    berr << cublasCreate_v2(&handle);

    cudaStream_t stream;
    err << cudaStreamCreate(&stream);

    berr << cublasSetStream_v2(handle, stream);

    double *dx;
    err << cudaMalloc(&dx, size * sizeof(float2));
    err << cudaMemcpyAsync(dx, a, size * sizeof(float2), cudaMemcpyHostToDevice, stream);
    berr << cublasScasum_v2(handle, n, reinterpret_cast<const cuComplex *>(dx), incx, res);
    return CudaAbstractTask(stream, handle) << dx;
}

CudaAbstractTask cu_dcasum(void *a, double *res, uint64_t n, uint64_t incx, uint64_t size) {
    CudaError err;
    CublasError berr;
    cublasHandle_t handle;
    berr << cublasCreate_v2(&handle);

    cudaStream_t stream;
    err << cudaStreamCreate(&stream);

    berr << cublasSetStream_v2(handle, stream);

    double *dx;
    err << cudaMalloc(&dx, size * sizeof(double2));
    err << cudaMemcpyAsync(dx, a, size * sizeof(double2), cudaMemcpyHostToDevice, stream);
    berr << cublasDzasum_v2(handle, n, reinterpret_cast<const cuDoubleComplex *>(dx), incx, res);
    return CudaAbstractTask(stream, handle) << dx;
}
