#ifndef SOKUDO_OPENCL_BLAS_LEVEL1_ADD_H
#define SOKUDO_OPENCL_BLAS_LEVEL1_ADD_H

#include <task.h>
#include <cstdint>

namespace sokudo::opencl::kernels::blas {
    /*
     * Can be used for sasum() and scasum()
     */
    CLTask *cl_sasum(const sokudo::DataBuffer<float> &x, const sokudo::DataValue<float> &res);
}

#endif
