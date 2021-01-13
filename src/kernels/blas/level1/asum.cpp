#include "asum.h"

#ifdef SOKUDO_CUDA

void detail(
        sokudo::CUDATask *task,
        const std::string &name,
        const std::string &x,
        const std::string &incx,
        const std::string &res
) {
    task->set(sokudo::KERNEL_BLAS_ASUM, name);
    task->params()["x"] = x;
    task->params()["incx"] = incx;
    task->params()["res"] = res;
}

sokudo::CUDATask *sokudo::kernels::blas::cuda_wrapper::asum::cuda_sasum(
        const sokudo::Buffer<float> &a,
        const sokudo::Value<uint64_t> &incx,
        const sokudo::Value<float> &res
) {
    auto task = cu_sasum(a.inner(), res.inner(), a.size(), incx.value());

    auto wtask = new CUDATask(task);
    detail(wtask, "BLAS_SASUM", a.name(), incx.name(), res.name());
    return wtask;
}

sokudo::CUDATask *sokudo::kernels::blas::cuda_wrapper::asum::cuda_dasum(
        const sokudo::Buffer<double> &a,
        const sokudo::Value<uint64_t> &incx,
        const sokudo::Value<double> &res
) {
    auto task = cu_dasum(a.inner(), res.inner(), a.size(), incx.value());

    auto wtask = new CUDATask(task);
    detail(wtask, "BLAS_DASUM", a.name(), incx.name(), res.name());
    return wtask;
}

sokudo::CUDATask *sokudo::kernels::blas::cuda_wrapper::asum::cuda_scasum(
        const sokudo::Buffer<float2> &a,
        const sokudo::Value<uint64_t> &incx,
        const sokudo::Value<float> &res
) {
    auto task = cu_scasum(a.inner(), res.inner(), a.size(), incx.value());

    auto wtask = new CUDATask(task);
    detail(wtask, "BLAS_SCASUM", a.name(), incx.name(), res.name());
    return wtask;
}

sokudo::CUDATask *sokudo::kernels::blas::cuda_wrapper::asum::cuda_dcasum(const sokudo::Buffer<double2> &a,
                                                                   const sokudo::Value<uint64_t> &incx,
                                                                   const sokudo::Value<double> &res) {
    auto task = cu_dcasum(a.inner(), res.inner(), a.size(), incx.value());

    auto wtask = new CUDATask(task);
    detail(wtask, "BLAS_DCASUM", a.name(), incx.name(), res.name());
    return wtask;
}

#endif