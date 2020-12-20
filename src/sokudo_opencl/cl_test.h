#ifndef SOKUDO_CL_TEST_H
#define SOKUDO_CL_TEST_H
#include "cl_helper.h"

namespace sokudo::opencl::kernels {
    void cl_platform_test();

    sokudo::opencl::CLTask cl_add_test(int *a, int *b, int n);
}

#endif
