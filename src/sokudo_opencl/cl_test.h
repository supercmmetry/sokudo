#ifndef SOKUDO_CL_TEST_H
#define SOKUDO_CL_TEST_H
#include "cl_helper.h"

namespace sokudo::opencl::kernels {
    void cl_platform_test();

    CLTask* cl_add_test(const sokudo::DataBuffer<int> &a, const sokudo::DataBuffer<int> &b);
}

#endif
