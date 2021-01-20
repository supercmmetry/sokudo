#ifndef SOKUDO_CUDA_BLAS_AXPY_H
#define SOKUDO_CUDA_BLAS_AXPY_H
#include "../../cuda_helper.h"

CudaAbstractTask cu_saxpy(float *alpha, float *x, uint64_t incx, float *y, uint64_t incy, uint64_t n);
CudaAbstractTask cu_daxpy(double *alpha, double *x, uint64_t incx, double *y, uint64_t incy, uint64_t n);
CudaAbstractTask cu_scaxpy(void *alpha, void *x, uint64_t incx, void *y, uint64_t incy, uint64_t n);
CudaAbstractTask cu_dcaxpy(void *alpha, void *x, uint64_t incx, void *y, uint64_t incy, uint64_t n);


#endif
