#ifndef SOKUDO_OPENCL_BLAS_AXPY_H
#define SOKUDO_OPENCL_BLAS_AXPY_H
#include <task.h>
#include <cstdint>

namespace sokudo::opencl::kernels::blas {
    CLTask *cl_saxpy(
            const sokudo::Buffer<float> &alpha,
            const sokudo::Buffer<float> &x,
            const sokudo::Value<uint64_t> &incx,
            const sokudo::Buffer<float> &y,
            const sokudo::Value<uint64_t> &incy,
            uint64_t wgs = 0,
            uint64_t stride = 0
    );

    CLTask *cl_daxpy(
            const sokudo::Buffer<double> &alpha,
            const sokudo::Buffer<double> &x,
            const sokudo::Value<uint64_t> &incx,
            const sokudo::Buffer<double> &y,
            const sokudo::Value<uint64_t> &incy,
            uint64_t wgs = 0,
            uint64_t stride = 0
    );

    CLTask *cl_scaxpy(
            const sokudo::Buffer<float2> &alpha,
            const sokudo::Buffer<float2> &x,
            const sokudo::Value<uint64_t> &incx,
            const sokudo::Buffer<float2> &y,
            const sokudo::Value<uint64_t> &incy,
            uint64_t wgs = 0,
            uint64_t stride = 0
    );

    CLTask *cl_dcaxpy(
            const sokudo::Buffer<double2> &alpha,
            const sokudo::Buffer<double2> &x,
            const sokudo::Value<uint64_t> &incx,
            const sokudo::Buffer<double2> &y,
            const sokudo::Value<uint64_t> &incy,
            uint64_t wgs = 0,
            uint64_t stride = 0
    );

    void register_axpy();
}
#endif
