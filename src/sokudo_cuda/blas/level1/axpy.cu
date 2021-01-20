#include <cublas.h>
#include "axpy.h"

#define MIN(a, b) ((a) < (b) ? (a) : (b))

CudaAbstractTask cu_saxpy(float *alpha, float *x, uint64_t incx, float *y, uint64_t incy, uint64_t n) {
    CudaError err;
    CublasError berr;
    cublasHandle_t handle;
    berr << cublasCreate_v2(&handle);

    cudaStream_t stream;
    err << cudaStreamCreate(&stream);

    berr << cublasSetStream_v2(handle, stream);

    float *dx, *dy;
    err << cudaMalloc(&dx, n * sizeof(float));
    err << cudaMalloc(&dy, n * sizeof(float));
    err << cudaMemcpyAsync(dx, x, n * sizeof(float), cudaMemcpyHostToDevice, stream);
    err << cudaMemcpyAsync(dy, y, n * sizeof(float), cudaMemcpyHostToDevice, stream);
    auto p1 = n / incx + (n % incx != 0);
    auto p2 = n / incy + (n % incy != 0);
    berr << cublasSaxpy_v2(handle, MIN(p1, p2), alpha, dx, incx, dy, incy);
    err << cudaMemcpyAsync(y, dy, n * sizeof(float), cudaMemcpyDeviceToHost, stream);
    return CudaAbstractTask(stream, handle) << dx << dy;
}

CudaAbstractTask cu_daxpy(double *alpha, double *x, uint64_t incx, double *y, uint64_t incy, uint64_t n) {
    CudaError err;
    CublasError berr;
    cublasHandle_t handle;
    berr << cublasCreate_v2(&handle);

    cudaStream_t stream;
    err << cudaStreamCreate(&stream);

    berr << cublasSetStream_v2(handle, stream);

    double *dx, *dy;
    err << cudaMalloc(&dx, n * sizeof(double));
    err << cudaMalloc(&dy, n * sizeof(double));
    err << cudaMemcpyAsync(dx, x, n * sizeof(double), cudaMemcpyHostToDevice, stream);
    err << cudaMemcpyAsync(dy, y, n * sizeof(double), cudaMemcpyHostToDevice, stream);
    auto p1 = n / incx + (n % incx != 0);
    auto p2 = n / incy + (n % incy != 0);
    berr << cublasDaxpy_v2(handle, MIN(p1, p2), alpha, dx, incx, dy, incy);
    err << cudaMemcpyAsync(y, dy, n * sizeof(double), cudaMemcpyDeviceToHost, stream);
    return CudaAbstractTask(stream, handle) << dx << dy;
}

CudaAbstractTask cu_scaxpy(void *alpha, void *x, uint64_t incx, void *y, uint64_t incy, uint64_t n) {
    CudaError err;
    CublasError berr;
    cublasHandle_t handle;
    berr << cublasCreate_v2(&handle);

    cudaStream_t stream;
    err << cudaStreamCreate(&stream);

    berr << cublasSetStream_v2(handle, stream);

    cuComplex *dx, *dy;
    err << cudaMalloc(&dx, n * sizeof(cuComplex));
    err << cudaMalloc(&dy, n * sizeof(cuComplex));
    err << cudaMemcpyAsync(dx, x, n * sizeof(cuComplex), cudaMemcpyHostToDevice, stream);
    err << cudaMemcpyAsync(dy, y, n * sizeof(cuComplex), cudaMemcpyHostToDevice, stream);
    auto p1 = n / incx + (n % incx != 0);
    auto p2 = n / incy + (n % incy != 0);
    berr << cublasCaxpy_v2(handle, MIN(p1, p2), (cuComplex*) alpha, dx, incx, dy, incy);
    err << cudaMemcpyAsync(y, dy, n * sizeof(cuComplex), cudaMemcpyDeviceToHost, stream);
    return CudaAbstractTask(stream, handle) << dx << dy;
}

CudaAbstractTask cu_dcaxpy(void *alpha, void *x, uint64_t incx, void *y, uint64_t incy, uint64_t n) {
    CudaError err;
    CublasError berr;
    cublasHandle_t handle;
    berr << cublasCreate_v2(&handle);

    cudaStream_t stream;
    err << cudaStreamCreate(&stream);

    berr << cublasSetStream_v2(handle, stream);

    cuDoubleComplex *dx, *dy;
    err << cudaMalloc(&dx, n * sizeof(cuDoubleComplex));
    err << cudaMalloc(&dy, n * sizeof(cuDoubleComplex));
    err << cudaMemcpyAsync(dx, x, n * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice, stream);
    err << cudaMemcpyAsync(dy, y, n * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice, stream);
    auto p1 = n / incx + (n % incx != 0);
    auto p2 = n / incy + (n % incy != 0);
    berr << cublasZaxpy_v2(handle, MIN(p1, p2), (cuDoubleComplex*) alpha, dx, incx, dy, incy);
    err << cudaMemcpyAsync(y, dy, n * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost, stream);
    return CudaAbstractTask(stream, handle) << dx << dy;
}
