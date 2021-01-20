#include <common.h>
#include "amax.h"

#ifdef SOKUDO_CUDA

void detail_amax_cu(
        sokudo::CUDATask *task,
        const std::string &name,
        const std::string &n,
        const std::string &x,
        const std::string &incx,
        const std::string &res
) {
    task->set(sokudo::KERNEL_BLAS_AMAX, name);
    task->params()["n"] = n;
    task->params()["x"] = x;
    task->params()["incx"] = incx;
    task->params()["res"] = res;
}

sokudo::CUDATask *sokudo::kernels::blas::cuda_wrapper::amax::cuda_samax(
        const sokudo::Value<uint64_t> &n,
        const sokudo::Buffer<float> &a,
        const sokudo::Value<uint64_t> &incx,
        const sokudo::Value<uint64_t> &res
) {
    auto task = cu_samax(a.inner(), res.inner(), n.value(), incx.value(), a.size());

    auto wtask = new CUDATask(task);
    detail_amax_cu(wtask, "BLAS_SAMAX", n.name(), a.name(), incx.name(), res.name());
    return wtask;
}

sokudo::CUDATask *sokudo::kernels::blas::cuda_wrapper::amax::cuda_damax(
        const sokudo::Value<uint64_t> &n,
        const sokudo::Buffer<double> &a,
        const sokudo::Value<uint64_t> &incx,
        const sokudo::Value<uint64_t> &res
) {
    auto task = cu_damax(a.inner(), res.inner(), n.value(), incx.value(), a.size());

    auto wtask = new CUDATask(task);
    detail_amax_cu(wtask, "BLAS_DAMAX", n.name(), a.name(), incx.name(), res.name());
    return wtask;
}

sokudo::CUDATask *sokudo::kernels::blas::cuda_wrapper::amax::cuda_scamax(
        const sokudo::Value<uint64_t> &n,
        const sokudo::Buffer<float2> &a,
        const sokudo::Value<uint64_t> &incx,
        const sokudo::Value<uint64_t> &res
) {
    auto task = cu_scamax(a.inner(), res.inner(), n.value(), incx.value(), a.size());

    auto wtask = new CUDATask(task);
    detail_amax_cu(wtask, "BLAS_SCAMAX", n.name(), a.name(), incx.name(), res.name());
    return wtask;
}

sokudo::CUDATask *sokudo::kernels::blas::cuda_wrapper::amax::cuda_dcamax(
        const sokudo::Value<uint64_t> &n,
        const sokudo::Buffer<double2> &a,
        const sokudo::Value<uint64_t> &incx,
        const sokudo::Value<uint64_t> &res
) {
    auto task = cu_dcamax(a.inner(), res.inner(), n.value(), incx.value(), a.size());

    auto wtask = new CUDATask(task);
    detail_amax_cu(wtask, "BLAS_DCAMAX", n.name(), a.name(), incx.name(), res.name());
    return wtask;
}

#endif