#ifndef SOKUDO_CUDA_BLAS_AMAX_H
#define SOKUDO_CUDA_BLAS_AMAX_H
#include "../../cuda_helper.h"

CudaAbstractTask cu_samax(float *a, uint64_t *res, uint64_t n, uint64_t incx, uint64_t size);
CudaAbstractTask cu_damax(double *a, uint64_t *res, uint64_t n, uint64_t incx, uint64_t size);
CudaAbstractTask cu_scamax(void *a, uint64_t *res, uint64_t n, uint64_t incx, uint64_t size);
CudaAbstractTask cu_dcamax(void *a, uint64_t *res, uint64_t n, uint64_t incx, uint64_t size);
#endif
