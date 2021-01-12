#include "asum.h"

#ifdef SOKUDO_CUDA
sokudo::CUDATask *sokudo::kernels::blas::cuda_wrapper::cuda_sasum(const sokudo::DataBuffer<float> &a,
                                                          const sokudo::DataValue<float> &res) {
    auto task = cu_sasum(a.inner(), res.inner(), a.size());
    return new CUDATask(task);
}
#endif