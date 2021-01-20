#include <cublas.h>
#include "amin.h"

CudaAbstractTask cu_samin(float *a, uint64_t *res, uint64_t n, uint64_t incx, uint64_t size) {
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
    int r = 0;
    berr << cublasIsamin_v2(handle, n, dx, incx, &r);
    *res = r;
    return CudaAbstractTask(stream, handle) << dx;
}

CudaAbstractTask cu_damin(double *a, uint64_t *res, uint64_t n, uint64_t incx, uint64_t size) {
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
    int r = 0;
    berr << cublasIdamin_v2(handle, n, dx, incx, &r);
    *res = r;
    return CudaAbstractTask(stream, handle) << dx;
}

CudaAbstractTask cu_scamin(void *a, uint64_t *res, uint64_t n, uint64_t incx, uint64_t size) {
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
    int r = 0;
    berr << cublasIcamin_v2(handle, n, reinterpret_cast<const cuComplex *>(dx), incx, &r);
    *res = r;
    return CudaAbstractTask(stream, handle) << dx;
}

CudaAbstractTask cu_dcamin(void *a, uint64_t *res, uint64_t n, uint64_t incx, uint64_t size) {
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
    int r = 0;
    berr << cublasIzamin_v2(handle, n, reinterpret_cast<const cuDoubleComplex *>(dx), incx, &r);
    *res = r;
    return CudaAbstractTask(stream, handle) << dx;
}
