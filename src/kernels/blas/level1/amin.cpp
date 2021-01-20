#include "amin.h"

#ifdef SOKUDO_CUDA
#include <sokudo_cuda/blas/level1/amin.h>

void detail_amin_cu(
        sokudo::CUDATask *task,
        const std::string &name,
        const std::string &n,
        const std::string &x,
        const std::string &incx,
        const std::string &res
) {
    task->set(sokudo::KERNEL_BLAS_AMIN, name);
    task->params()["n"] = n;
    task->params()["x"] = x;
    task->params()["incx"] = incx;
    task->params()["res"] = res;
}

sokudo::CUDATask *sokudo::kernels::blas::cuda_wrapper::amin::cuda_samin(
        const sokudo::Value<uint64_t> &n,
        const sokudo::Buffer<float> &a,
        const sokudo::Value<uint64_t> &incx,
        const sokudo::Value<uint64_t> &res
) {
    auto task = cu_samin(a.inner(), res.inner(), n.value(), incx.value(), a.size());

    auto wtask = new CUDATask(task);
    detail_amin_cu(wtask, "BLAS_SAMIN", n.name(), a.name(), incx.name(), res.name());
    return wtask;
}

sokudo::CUDATask *sokudo::kernels::blas::cuda_wrapper::amin::cuda_damin(
        const sokudo::Value<uint64_t> &n,
        const sokudo::Buffer<double> &a,
        const sokudo::Value<uint64_t> &incx,
        const sokudo::Value<uint64_t> &res
) {
    auto task = cu_damin(a.inner(), res.inner(), n.value(), incx.value(), a.size());

    auto wtask = new CUDATask(task);
    detail_amin_cu(wtask, "BLAS_DAMIN", n.name(), a.name(), incx.name(), res.name());
    return wtask;
}

sokudo::CUDATask *sokudo::kernels::blas::cuda_wrapper::amin::cuda_scamin(
        const sokudo::Value<uint64_t> &n,
        const sokudo::Buffer<float2> &a,
        const sokudo::Value<uint64_t> &incx,
        const sokudo::Value<uint64_t> &res
) {
    auto task = cu_scamin(a.inner(), res.inner(), n.value(), incx.value(), a.size());

    auto wtask = new CUDATask(task);
    detail_amin_cu(wtask, "BLAS_SCAMIN", n.name(), a.name(), incx.name(), res.name());
    return wtask;
}

sokudo::CUDATask *sokudo::kernels::blas::cuda_wrapper::amin::cuda_dcamin(
        const sokudo::Value<uint64_t> &n,
        const sokudo::Buffer<double2> &a,
        const sokudo::Value<uint64_t> &incx,
        const sokudo::Value<uint64_t> &res
) {
    auto task = cu_dcamin(a.inner(), res.inner(), n.value(), incx.value(), a.size());

    auto wtask = new CUDATask(task);
    detail_amin_cu(wtask, "BLAS_DCAMIN", n.name(), a.name(), incx.name(), res.name());
    return wtask;
}

#endif