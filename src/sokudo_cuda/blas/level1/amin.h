#ifndef SOKUDO_CUDA_BLAS_AMIN_H
#define SOKUDO_CUDA_BLAS_AMIN_H
#include "../../cuda_helper.h"

CudaAbstractTask cu_samin(float *a, uint64_t *res, uint64_t n, uint64_t incx);
CudaAbstractTask cu_damin(double *a, uint64_t *res, uint64_t n, uint64_t incx);
CudaAbstractTask cu_scamin(void *a, uint64_t *res, uint64_t n, uint64_t incx);
CudaAbstractTask cu_dcamin(void *a, uint64_t *res, uint64_t n, uint64_t incx);
#endif
