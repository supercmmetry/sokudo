#ifndef SOKUDO_OPENCL_BLAS_AMIN_H
#define SOKUDO_OPENCL_BLAS_AMIN_H

#include <task.h>
#include <cstdint>

namespace sokudo::opencl::kernels::blas {
    CLTask *cl_samin(
            const sokudo::Value<uint64_t> &n,
            const sokudo::Buffer<float> &x,
            const sokudo::Value<uint64_t> &incx,
            const sokudo::Value<uint64_t> &res,
            uint64_t wgs = 0,
            uint64_t stride = 0
    );

    CLTask *cl_damin(
            const sokudo::Value<uint64_t> &n,
            const sokudo::Buffer<double> &x,
            const sokudo::Value<uint64_t> &incx,
            const sokudo::Value<uint64_t> &res,
            uint64_t wgs = 0,
            uint64_t stride = 0
    );

    CLTask *cl_scamin(
            const sokudo::Value<uint64_t> &n,
            const sokudo::Buffer<float2> &x,
            const sokudo::Value<uint64_t> &incx,
            const sokudo::Value<uint64_t> &res,
            uint64_t wgs = 0,
            uint64_t stride = 0
    );

    CLTask *cl_dcamin(
            const sokudo::Value<uint64_t> &n,
            const sokudo::Buffer<double2> &x,
            const sokudo::Value<uint64_t> &incx,
            const sokudo::Value<uint64_t> &res,
            uint64_t wgs = 0,
            uint64_t stride = 0
    );

    void register_amin();
}

#endif
