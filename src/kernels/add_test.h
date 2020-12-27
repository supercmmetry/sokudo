#ifndef SOKUDO_ADD_TEST_H
#define SOKUDO_ADD_TEST_H

#include <task.h>

#ifdef SOKUDO_CUDA

#include <sokudo_cuda/cuda_test.h>

#endif

#ifdef SOKUDO_OPENCL

#include <sokudo_opencl/cl_test.h>

#endif

namespace sokudo::kernels {
#ifdef SOKUDO_CUDA
    namespace cuda_wrappers {
        sokudo::CUDATask *add_test(const sokudo::DataBuffer<int> &a, const sokudo::DataBuffer<int> &b);
    }
#endif

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
                    return dynamic_cast<Task *>(cuda_wrappers::add_test(a, b));
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
