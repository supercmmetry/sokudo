#include "asum.h"

#ifdef SOKUDO_CUDA

sokudo::CUDATask *sokudo::kernels::blas::cuda_wrapper::cuda_sasum(
        const sokudo::DataBuffer<float> &a,
        const sokudo::DataValue<uint64_t> &incx,
        const sokudo::DataValue<float> &res
) {
    auto task = cu_sasum(a.inner(), res.inner(), a.size(), incx.value());
    return new CUDATask(task);
}

sokudo::CUDATask *sokudo::kernels::blas::cuda_wrapper::cuda_dasum(
        const sokudo::DataBuffer<double> &a,
        const sokudo::DataValue<uint64_t> &incx,
        const sokudo::DataValue<double> &res
) {
    auto task = cu_dasum(a.inner(), res.inner(), a.size(), incx.value());
    return new CUDATask(task);
}

#endif