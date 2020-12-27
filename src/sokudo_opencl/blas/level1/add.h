#ifndef SOKUDO_ADD_H
#define SOKUDO_ADD_H

#include <task.h>
#include <cstdint>

namespace sokudo::opencl::kernels::blas {
    /*
     * Can be used for sasum() and scasum()
     */
    CLTask *cl_math_add_real(
            const sokudo::DataBuffer<int64_t> &a,
            const sokudo::DataBuffer<int64_t> &b,
            const sokudo::DataBuffer<int64_t> &c
    );
}

#endif
