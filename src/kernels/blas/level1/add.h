#ifndef SOKUDO_BLAS_LEVEL1_ADD_H
#define SOKUDO_BLAS_LEVEL1_ADD_H
#include <task.h>

#ifdef SOKUDO_CUDA

#include <sokudo_cuda/cuda_test.h>

#endif

#ifdef SOKUDO_OPENCL

#include <sokudo_opencl/blas/level1/add.h>

#endif

namespace sokudo::kernels::blas {
#ifdef SOKUDO_CUDA

#endif

    template<TaskExecutor executor>
    class AddInt {
    public:
        AddInt() = default;

        Task *operator()(const sokudo::DataBuffer<int64_t> &a, const sokudo::DataBuffer<int64_t> &b, const sokudo::DataBuffer<int64_t> &c) const {
            TaskExecutor fallback_executor = get_fallback_executor(executor);

            switch (fallback_executor) {
                case CPU:
                    throw sokudo::errors::ResolutionException("CPU implementation not found");
                case OPENCL:
#ifdef SOKUDO_OPENCL
                    return dynamic_cast<Task *>(sokudo::opencl::kernels::blas::cl_math_add_int(a, b, c));
#else
                    throw sokudo::errors::ResolutionException("OpenCL implementation not found");
#endif
                case CUDA:
#ifdef SOKUDO_CUDA
                    throw sokudo::errors::ResolutionException("CUDA implementation not found");
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
