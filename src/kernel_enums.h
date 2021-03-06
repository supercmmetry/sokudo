#ifndef SOKUDO_KERNEL_ENUMS_H
#define SOKUDO_KERNEL_ENUMS_H

namespace sokudo {
    enum Kernel {
        KERNEL_UNDEFINED,
        KERNEL_BLAS_ASUM,
        KERNEL_BLAS_AMAX,
        KERNEL_BLAS_AMIN,
        KERNEL_BLAS_AXPY
    };
}

#endif
