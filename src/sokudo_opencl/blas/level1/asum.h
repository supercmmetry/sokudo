#ifndef SOKUDO_OPENCL_BLAS_LEVEL1_ADD_H
#define SOKUDO_OPENCL_BLAS_LEVEL1_ADD_H

#include <task.h>
#include <cstdint>

namespace sokudo::opencl::kernels::blas {
    /*
     * Can be used for sasum() and scasum()
     */
    CLTask *cl_sasum(
            const sokudo::DataBuffer<float> &x,
            const sokudo::DataValue<uint64_t> &incx,
            const sokudo::DataValue<float> &res,
            uint64_t wgs = 0,
            uint64_t stride = 0
    );

    CLTask *cl_dasum(
            const sokudo::DataBuffer<double> &x,
            const sokudo::DataValue<uint64_t> &incx,
            const sokudo::DataValue<double> &res,
            uint64_t wgs = 0,
            uint64_t stride = 0
    );

    void register_sasum();
    void register_dasum();
}

#endif
