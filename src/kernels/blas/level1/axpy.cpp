#include "axpy.h"

#ifdef SOKUDO_CUDA

#include <sokudo_cuda/blas/level1/axpy.h>

void detail_axpy_cu(
        sokudo::CUDATask *task,
        const std::string &name,
        const std::string &n,
        const std::string &alpha,
        const std::string &x,
        const std::string &incx,
        const std::string &y,
        const std::string &incy
) {
    task->set(sokudo::KERNEL_BLAS_AXPY, name);
    task->params()["n"] = n;
    task->params()["alpha"] = alpha;
    task->params()["x"] = x;
    task->params()["incx"] = incx;
    task->params()["y"] = y;
    task->params()["incy"] = incy;
}

sokudo::CUDATask *sokudo::kernels::blas::cuda_wrapper::axpy::cuda_saxpy(
        const sokudo::Value<uint64_t> &n,
        const sokudo::Value<float> &alpha,
        const sokudo::Buffer<float> &x,
        const sokudo::Value<uint64_t> &incx,
        const sokudo::Buffer<float> &y,
        const sokudo::Value<uint64_t> &incy
) {
    auto task = cu_saxpy(alpha.inner(), x.inner(), incx.value(), y.inner(), incy.value(), n.value(), x.size(),
                         y.size());

    auto wtask = new CUDATask(task);
    detail_axpy_cu(wtask, "BLAS_SAXPY", n.name(), alpha.name(), x.name(), incx.name(), y.name(), incy.name());
    return wtask;
}

sokudo::CUDATask *sokudo::kernels::blas::cuda_wrapper::axpy::cuda_daxpy(
        const sokudo::Value<uint64_t> &n,
        const sokudo::Value<double> &alpha,
        const sokudo::Buffer<double> &x,
        const sokudo::Value<uint64_t> &incx,
        const sokudo::Buffer<double> &y,
        const sokudo::Value<uint64_t> &incy
) {
    auto task = cu_daxpy(alpha.inner(), x.inner(), incx.value(), y.inner(), incy.value(), n.value(), x.size(),
                         y.size());

    auto wtask = new CUDATask(task);
    detail_axpy_cu(wtask, "BLAS_DAXPY", n.name(), alpha.name(), x.name(), incx.name(), y.name(), incy.name());
    return wtask;
}

sokudo::CUDATask *sokudo::kernels::blas::cuda_wrapper::axpy::cuda_scaxpy(
        const sokudo::Value<uint64_t> &n,
        const sokudo::Value<float2> &alpha,
        const sokudo::Buffer<float2> &x,
        const sokudo::Value<uint64_t> &incx,
        const sokudo::Buffer<float2> &y,
        const sokudo::Value<uint64_t> &incy
) {
    auto task = cu_scaxpy(alpha.inner(), x.inner(), incx.value(), y.inner(), incy.value(), n.value(), x.size(),
                          y.size());

    auto wtask = new CUDATask(task);
    detail_axpy_cu(wtask, "BLAS_SCAXPY", n.name(), alpha.name(), x.name(), incx.name(), y.name(), incy.name());
    return wtask;
}

sokudo::CUDATask *sokudo::kernels::blas::cuda_wrapper::axpy::cuda_dcaxpy(
        const sokudo::Value<uint64_t> &n,
        const sokudo::Value<double2> &alpha,
        const sokudo::Buffer<double2> &x,
        const sokudo::Value<uint64_t> &incx,
        const sokudo::Buffer<double2> &y,
        const sokudo::Value<uint64_t> &incy
) {
    auto task = cu_dcaxpy(alpha.inner(), x.inner(), incx.value(), y.inner(), incy.value(), n.value(), x.size(),
                          y.size());

    auto wtask = new CUDATask(task);
    detail_axpy_cu(wtask, "BLAS_DCAXPY", n.name(), alpha.name(), x.name(), incx.name(), y.name(), incy.name());
    return wtask;
}

#endif