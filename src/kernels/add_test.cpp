#include "add_test.h"

sokudo::CUDATask* sokudo::kernels::cuda_wrappers::add_test(const sokudo::DataBuffer<int> &a, const sokudo::DataBuffer<int> &b) {
    CudaAbstractTask abstract_task = cu_add_test(a.inner(), b.inner(), a.size());
    return new CUDATask(abstract_task);
}
