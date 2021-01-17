#ifndef SOKUDO_OPENCL_BLAS_ASUM_H
#define SOKUDO_OPENCL_BLAS_ASUM_H

#include <task.h>
#include <cstdint>

namespace sokudo::opencl::kernels::blas {
    CLTask *cl_sasum(
            const sokudo::Buffer<float> &x,
            const sokudo::Value<uint64_t> &incx,
            const sokudo::Value<float> &res,
            uint64_t wgs = 0,
            uint64_t stride = 0
    );

    CLTask *cl_dasum(
            const sokudo::Buffer<double> &x,
            const sokudo::Value<uint64_t> &incx,
            const sokudo::Value<double> &res,
            uint64_t wgs = 0,
            uint64_t stride = 0
    );

    CLTask *cl_scasum(
            const sokudo::Buffer<float2> &x,
            const sokudo::Value<uint64_t> &incx,
            const sokudo::Value<float> &res,
            uint64_t wgs = 0,
            uint64_t stride = 0
    );

    CLTask *cl_dcasum(
            const sokudo::Buffer<double2> &x,
            const sokudo::Value<uint64_t> &incx,
            const sokudo::Value<double> &res,
            uint64_t wgs = 0,
            uint64_t stride = 0
    );

    void register_asum();
}

#endif
