#ifndef SOKUDO_OPENCL_BLAS_AMAX_H
#define SOKUDO_OPENCL_BLAS_AMAX_H

#include <task.h>
#include <cstdint>

namespace sokudo::opencl::kernels::blas {
    CLTask *cl_samax(
            const sokudo::Value<uint64_t> &n,
            const sokudo::Buffer<float> &x,
            const sokudo::Value<uint64_t> &incx,
            const sokudo::Value<uint64_t> &res,
            uint64_t wgs = 0,
            uint64_t stride = 0
    );

    CLTask *cl_damax(
            const sokudo::Value<uint64_t> &n,
            const sokudo::Buffer<double> &x,
            const sokudo::Value<uint64_t> &incx,
            const sokudo::Value<uint64_t> &res,
            uint64_t wgs = 0,
            uint64_t stride = 0
    );

    CLTask *cl_scamax(
            const sokudo::Value<uint64_t> &n,
            const sokudo::Buffer<float2> &x,
            const sokudo::Value<uint64_t> &incx,
            const sokudo::Value<uint64_t> &res,
            uint64_t wgs = 0,
            uint64_t stride = 0
    );

    CLTask *cl_dcamax(
            const sokudo::Value<uint64_t> &n,
            const sokudo::Buffer<double2> &x,
            const sokudo::Value<uint64_t> &incx,
            const sokudo::Value<uint64_t> &res,
            uint64_t wgs = 0,
            uint64_t stride = 0
    );

    void register_amax();
}

#endif
