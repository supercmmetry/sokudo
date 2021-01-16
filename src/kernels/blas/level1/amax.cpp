#include "amax.h"

#ifdef SOKUDO_CUDA

void detail_amax_cu(
        sokudo::CUDATask *task,
        const std::string &name,
        const std::string &x,
        const std::string &incx,
        const std::string &res
) {
    task->set(sokudo::KERNEL_BLAS_AMAX, name);
    task->params()["x"] = x;
    task->params()["incx"] = incx;
    task->params()["res"] = res;
}

sokudo::CUDATask *sokudo::kernels::blas::cuda_wrapper::amax::cuda_samax(
        const sokudo::Buffer<float> &a,
        const sokudo::Value<uint64_t> &incx,
        const sokudo::Value<uint64_t> &res
) {
    auto task = cu_samax(a.inner(), res.inner(), a.size(), incx.value());

    auto wtask = new CUDATask(task);
    detail_amax_cu(wtask, "BLAS_SAMAX", a.name(), incx.name(), res.name());
    return wtask;
}

sokudo::CUDATask *sokudo::kernels::blas::cuda_wrapper::amax::cuda_damax(
        const sokudo::Buffer<double> &a,
        const sokudo::Value<uint64_t> &incx,
        const sokudo::Value<uint64_t> &res
) {
    auto task = cu_damax(a.inner(), res.inner(), a.size(), incx.value());

    auto wtask = new CUDATask(task);
    detail_amax_cu(wtask, "BLAS_DAMAX", a.name(), incx.name(), res.name());
    return wtask;
}

sokudo::CUDATask *sokudo::kernels::blas::cuda_wrapper::amax::cuda_scamax(
        const sokudo::Buffer<float2> &a,
        const sokudo::Value<uint64_t> &incx,
        const sokudo::Value<uint64_t> &res
) {
    auto task = cu_scamax(a.inner(), res.inner(), a.size(), incx.value());

    auto wtask = new CUDATask(task);
    detail_amax_cu(wtask, "BLAS_SCAMAX", a.name(), incx.name(), res.name());
    return wtask;
}

sokudo::CUDATask *sokudo::kernels::blas::cuda_wrapper::amax::cuda_dcamax(const sokudo::Buffer<double2> &a,
                                                                         const sokudo::Value<uint64_t> &incx,
                                                                         const sokudo::Value<uint64_t> &res) {
    auto task = cu_dcamax(a.inner(), res.inner(), a.size(), incx.value());

    auto wtask = new CUDATask(task);
    detail_amax_cu(wtask, "BLAS_DCAMAX", a.name(), incx.name(), res.name());
    return wtask;
}

#endif