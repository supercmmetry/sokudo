#include "amin.h"

#ifdef SOKUDO_CUDA

void detail_amin_cu(
        sokudo::CUDATask *task,
        const std::string &name,
        const std::string &x,
        const std::string &incx,
        const std::string &res
) {
    task->set(sokudo::KERNEL_BLAS_AMIN, name);
    task->params()["x"] = x;
    task->params()["incx"] = incx;
    task->params()["res"] = res;
}

sokudo::CUDATask *sokudo::kernels::blas::cuda_wrapper::amin::cuda_samin(
        const sokudo::Buffer<float> &a,
        const sokudo::Value<uint64_t> &incx,
        const sokudo::Value<uint64_t> &res
) {
    auto task = cu_samin(a.inner(), res.inner(), a.size(), incx.value());

    auto wtask = new CUDATask(task);
    detail_amin_cu(wtask, "BLAS_SAMIN", a.name(), incx.name(), res.name());
    return wtask;
}

sokudo::CUDATask *sokudo::kernels::blas::cuda_wrapper::amin::cuda_damin(
        const sokudo::Buffer<double> &a,
        const sokudo::Value<uint64_t> &incx,
        const sokudo::Value<uint64_t> &res
) {
    auto task = cu_damin(a.inner(), res.inner(), a.size(), incx.value());

    auto wtask = new CUDATask(task);
    detail_amin_cu(wtask, "BLAS_DAMIN", a.name(), incx.name(), res.name());
    return wtask;
}

sokudo::CUDATask *sokudo::kernels::blas::cuda_wrapper::amin::cuda_scamin(
        const sokudo::Buffer<float2> &a,
        const sokudo::Value<uint64_t> &incx,
        const sokudo::Value<uint64_t> &res
) {
    auto task = cu_scamin(a.inner(), res.inner(), a.size(), incx.value());

    auto wtask = new CUDATask(task);
    detail_amin_cu(wtask, "BLAS_SCAMIN", a.name(), incx.name(), res.name());
    return wtask;
}

sokudo::CUDATask *sokudo::kernels::blas::cuda_wrapper::amin::cuda_dcamin(const sokudo::Buffer<double2> &a,
                                                                         const sokudo::Value<uint64_t> &incx,
                                                                         const sokudo::Value<uint64_t> &res) {
    auto task = cu_dcamin(a.inner(), res.inner(), a.size(), incx.value());

    auto wtask = new CUDATask(task);
    detail_amin_cu(wtask, "BLAS_DCAMIN", a.name(), incx.name(), res.name());
    return wtask;
}

#endif