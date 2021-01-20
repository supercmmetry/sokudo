#include <cublas.h>
#include "axpy.h"

CudaAbstractTask
cu_saxpy(float *alpha, float *x, uint64_t incx, float *y, uint64_t incy, uint64_t n, uint64_t size_x, uint64_t size_y) {
    CudaError err;
    CublasError berr;
    cublasHandle_t handle;
    berr << cublasCreate_v2(&handle);

    cudaStream_t stream;
    err << cudaStreamCreate(&stream);

    berr << cublasSetStream_v2(handle, stream);

    float *dx, *dy;
    err << cudaMalloc(&dx, size_x * sizeof(float));
    err << cudaMalloc(&dy, size_y * sizeof(float));
    err << cudaMemcpyAsync(dx, x, size_x * sizeof(float), cudaMemcpyHostToDevice, stream);
    err << cudaMemcpyAsync(dy, y, size_y * sizeof(float), cudaMemcpyHostToDevice, stream);
    berr << cublasSaxpy_v2(handle, n, alpha, dx, incx, dy, incy);
    err << cudaMemcpyAsync(y, dy, size_y * sizeof(float), cudaMemcpyDeviceToHost, stream);
    return CudaAbstractTask(stream, handle) << dx << dy;
}

CudaAbstractTask
cu_daxpy(double *alpha, double *x, uint64_t incx, double *y, uint64_t incy, uint64_t n, uint64_t size_x,
         uint64_t size_y) {
    CudaError err;
    CublasError berr;
    cublasHandle_t handle;
    berr << cublasCreate_v2(&handle);

    cudaStream_t stream;
    err << cudaStreamCreate(&stream);

    berr << cublasSetStream_v2(handle, stream);

    double *dx, *dy;
    err << cudaMalloc(&dx, size_x * sizeof(double));
    err << cudaMalloc(&dy, size_y * sizeof(double));
    err << cudaMemcpyAsync(dx, x, size_x * sizeof(double), cudaMemcpyHostToDevice, stream);
    err << cudaMemcpyAsync(dy, y, size_y * sizeof(double), cudaMemcpyHostToDevice, stream);
    berr << cublasDaxpy_v2(handle, n, alpha, dx, incx, dy, incy);
    err << cudaMemcpyAsync(y, dy, size_y * sizeof(double), cudaMemcpyDeviceToHost, stream);
    return CudaAbstractTask(stream, handle) << dx << dy;
}

CudaAbstractTask
cu_scaxpy(void *alpha, void *x, uint64_t incx, void *y, uint64_t incy, uint64_t n, uint64_t size_x, uint64_t size_y) {
    CudaError err;
    CublasError berr;
    cublasHandle_t handle;
    berr << cublasCreate_v2(&handle);

    cudaStream_t stream;
    err << cudaStreamCreate(&stream);

    berr << cublasSetStream_v2(handle, stream);

    cuComplex *dx, *dy;
    err << cudaMalloc(&dx, size_x * sizeof(cuComplex));
    err << cudaMalloc(&dy, size_y * sizeof(cuComplex));
    err << cudaMemcpyAsync(dx, x, size_x * sizeof(cuComplex), cudaMemcpyHostToDevice, stream);
    err << cudaMemcpyAsync(dy, y, size_y * sizeof(cuComplex), cudaMemcpyHostToDevice, stream);
    berr << cublasCaxpy_v2(handle, n, (cuComplex *) alpha, dx, incx, dy, incy);
    err << cudaMemcpyAsync(y, dy, size_y * sizeof(cuComplex), cudaMemcpyDeviceToHost, stream);
    return CudaAbstractTask(stream, handle) << dx << dy;
}

CudaAbstractTask
cu_dcaxpy(void *alpha, void *x, uint64_t incx, void *y, uint64_t incy, uint64_t n, uint64_t size_x, uint64_t size_y) {
    CudaError err;
    CublasError berr;
    cublasHandle_t handle;
    berr << cublasCreate_v2(&handle);

    cudaStream_t stream;
    err << cudaStreamCreate(&stream);

    berr << cublasSetStream_v2(handle, stream);

    cuDoubleComplex *dx, *dy;
    err << cudaMalloc(&dx, size_x * sizeof(cuDoubleComplex));
    err << cudaMalloc(&dy, size_y * sizeof(cuDoubleComplex));
    err << cudaMemcpyAsync(dx, x, size_x * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice, stream);
    err << cudaMemcpyAsync(dy, y, size_y * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice, stream);
    berr << cublasZaxpy_v2(handle, n, (cuDoubleComplex *) alpha, dx, incx, dy, incy);
    err << cudaMemcpyAsync(y, dy, size_y * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost, stream);
    return CudaAbstractTask(stream, handle) << dx << dy;
}
