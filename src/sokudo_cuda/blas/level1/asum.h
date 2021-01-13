#ifndef SOKUDO_ASUM_H
#define SOKUDO_ASUM_H
#include "../../cuda_helper.h"

CudaAbstractTask cu_sasum(float *a, float *res, uint64_t n, uint64_t incx);
CudaAbstractTask cu_dasum(double *a, double *res, uint64_t n, uint64_t incx);
CudaAbstractTask cu_scasum(float2 *a, float *res, uint64_t n, uint64_t incx);
CudaAbstractTask cu_dcasum(double2 *a, double *res, uint64_t n, uint64_t incx);
#endif
