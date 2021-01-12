#ifndef SOKUDO_ASUM_H
#define SOKUDO_ASUM_H
#include "../../cuda_helper.h"

CudaAbstractTask cu_sasum(float *a, float *res, uint64_t n);
#endif
