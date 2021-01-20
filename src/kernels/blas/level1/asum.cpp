#include "asum.h"
#include <common.h>

#ifdef SOKUDO_CUDA

void detail_asum_cu(
        sokudo::CUDATask *task,
        const std::string &name,
        const std::string &n,
        const std::string &x,
        const std::string &incx,
        const std::string &res
) {
    task->set(sokudo::KERNEL_BLAS_ASUM, name);
    task->params()["n"] = n;
    task->params()["x"] = x;
    task->params()["incx"] = incx;
    task->params()["res"] = res;
}

sokudo::CUDATask *sokudo::kernels::blas::cuda_wrapper::asum::cuda_sasum(
        const sokudo::Value<uint64_t> &n,
        const sokudo::Buffer<float> &a,
        const sokudo::Value<uint64_t> &incx,
        const sokudo::Value<float> &res
) {
    auto task = cu_sasum(a.inner(), res.inner(), n.value(), incx.value(), a.size());

    auto wtask = new CUDATask(task);
    detail_asum_cu(wtask, "BLAS_SASUM", n.name(), a.name(), incx.name(), res.name());
    return wtask;
}

sokudo::CUDATask *sokudo::kernels::blas::cuda_wrapper::asum::cuda_dasum(
        const sokudo::Value<uint64_t> &n,
        const sokudo::Buffer<double> &a,
        const sokudo::Value<uint64_t> &incx,
        const sokudo::Value<double> &res
) {
    auto task = cu_dasum(a.inner(), res.inner(), n.value(), incx.value(), a.size());

    auto wtask = new CUDATask(task);
    detail_asum_cu(wtask, "BLAS_DASUM", n.name(), a.name(), incx.name(), res.name());
    return wtask;
}

sokudo::CUDATask *sokudo::kernels::blas::cuda_wrapper::asum::cuda_scasum(
        const sokudo::Value<uint64_t> &n,
        const sokudo::Buffer<float2> &a,
        const sokudo::Value<uint64_t> &incx,
        const sokudo::Value<float> &res
) {
    auto task = cu_scasum(a.inner(), res.inner(), n.value(), incx.value(), a.size());

    auto wtask = new CUDATask(task);
    detail_asum_cu(wtask, "BLAS_SCASUM", n.name(), a.name(), incx.name(), res.name());
    return wtask;
}

sokudo::CUDATask *sokudo::kernels::blas::cuda_wrapper::asum::cuda_dcasum(
        const sokudo::Value<uint64_t> &n,
        const sokudo::Buffer<double2> &a,
        const sokudo::Value<uint64_t> &incx,
        const sokudo::Value<double> &res
) {
    auto task = cu_dcasum(a.inner(), res.inner(), n.value(), incx.value(), a.size());

    auto wtask = new CUDATask(task);
    detail_asum_cu(wtask, "BLAS_DCASUM", n.name(), a.name(), incx.name(), res.name());
    return wtask;
}

#endif