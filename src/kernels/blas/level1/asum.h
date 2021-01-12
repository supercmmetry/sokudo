#ifndef SOKUDO_BLAS_LEVEL1_ASUM_H
#define SOKUDO_BLAS_LEVEL1_ASUM_H
#include <task.h>

#ifdef SOKUDO_CUDA

#include <sokudo_cuda/blas/level1/asum.h>

#endif

#ifdef SOKUDO_OPENCL

#include <sokudo_opencl/blas/level1/asum.h>

#endif

namespace sokudo::kernels::blas {
#ifdef SOKUDO_CUDA
    namespace cuda_wrapper {
        CUDATask *cuda_sasum(const sokudo::DataBuffer<float> &a, const sokudo::DataValue<float> &res);
    }
#endif

    template<TaskExecutor executor>
    class Sasum {
    public:
        Sasum() = default;

        Task *operator()(const sokudo::DataBuffer<float> &a, const sokudo::DataValue<float> &res) const {
            TaskExecutor fallback_executor = get_fallback_executor(executor);

            switch (fallback_executor) {
                case CPU:
                    throw sokudo::errors::ResolutionException("CPU implementation not found");
                case OPENCL:
#ifdef SOKUDO_OPENCL
                    return dynamic_cast<Task *>(sokudo::opencl::kernels::blas::cl_sasum(a, res));
#else
                    throw sokudo::errors::ResolutionException("OpenCL implementation not found");
#endif
                case CUDA:
#ifdef SOKUDO_CUDA
                    return dynamic_cast<Task *>(cuda_wrapper::cuda_sasum(a, res));
#else
                    throw sokudo::errors::ResolutionException("CUDA implementation not found");
#endif
                default:
                    throw sokudo::errors::InvalidOperationException("Undefined task executor");
            }
        }
    };
}
#endif
