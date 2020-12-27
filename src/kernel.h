#ifndef SOKUDO_KERNEL_H
#define SOKUDO_KERNEL_H

#include <kernel_enums.h>
#include <task.h>

#ifdef SOKUDO_CUDA

#include <sokudo_cuda/cuda_test.h>

#endif

#ifdef SOKUDO_OPENCL

#include <sokudo_opencl/cl_test.h>

#endif

namespace sokudo::kernels {
#ifdef SOKUDO_CUDA
    namespace cuda {
        sokudo::CUDATask *add_test(const sokudo::DataBuffer<int> &a, const sokudo::DataBuffer<int> &b);
    }
#endif

    inline TaskExecutor get_fallback_executor(TaskExecutor executor) {
        auto fallback_executor = executor;
#if !defined(SOKUDO_CUDA)
#if defined(SOKUDO_OPENCL)
        if (fallback_executor == CUDA) {
            fallback_executor = OPENCL;
        }
#else
        if (fallback_executor == CUDA) {
            fallback_executor = CPU;
        }
#endif
#endif

#if !defined(SOKUDO_OPENCL)
#if defined(SOKUDO_CUDA)
        if (fallback_executor == OPENCL) {
            fallback_executor = CUDA;
        }
#else
        if (fallback_executor == OPENCL) {
            fallback_executor = CPU;
        }
#endif
#endif

        return fallback_executor;
    }

    template<TaskExecutor executor>
    class AddTest {
    public:
        AddTest() = default;

        Task *operator()(const sokudo::DataBuffer<int> &a, const sokudo::DataBuffer<int> &b) const {
            TaskExecutor fallback_executor = get_fallback_executor(executor);

            switch (fallback_executor) {
                case CPU:
                    throw sokudo::errors::ResolutionException("CPU implementation not found");
                case OPENCL:
#ifdef SOKUDO_OPENCL
                    return dynamic_cast<Task *>(sokudo::opencl::kernels::cl_add_test(a, b));
#else
                    throw sokudo::errors::ResolutionException("OpenCL implementation not found");
#endif
                case CUDA:
#ifdef SOKUDO_CUDA
                    return dynamic_cast<Task *>(cuda::add_test(a, b));
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
