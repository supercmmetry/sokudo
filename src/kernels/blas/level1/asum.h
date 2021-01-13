#ifndef SOKUDO_BLAS_LEVEL1_ASUM_H
#define SOKUDO_BLAS_LEVEL1_ASUM_H

#include <task.h>
#include <types.h>

#ifdef SOKUDO_CUDA

#include <sokudo_cuda/blas/level1/asum.h>

#endif

#ifdef SOKUDO_OPENCL

#include <sokudo_opencl/blas/level1/asum.h>

#endif

namespace sokudo::kernels::blas {
#ifdef SOKUDO_CUDA
    namespace cuda_wrapper::asum {
        CUDATask *cuda_sasum(const sokudo::Buffer<float> &a, const sokudo::Value<uint64_t> &incx,
                             const sokudo::Value<float> &res);

        CUDATask *cuda_dasum(const sokudo::Buffer<double> &a, const sokudo::Value<uint64_t> &incx,
                             const sokudo::Value<double> &res);

        CUDATask *cuda_scasum(const sokudo::Buffer<float2> &a, const sokudo::Value<uint64_t> &incx,
                              const sokudo::Value<float> &res);

        CUDATask *cuda_dcasum(const sokudo::Buffer<double2> &a, const sokudo::Value<uint64_t> &incx,
                              const sokudo::Value<double> &res);
    }
#endif

    template<TaskExecutor executor>
    class Asum {
    public:
        Asum() = default;

        // sasum
        Task *operator()(
                const sokudo::Buffer<float> &a,
                const sokudo::Value<uint64_t> &incx,
                const sokudo::Value<float> &res
        ) const {
            TaskExecutor fallback_executor = get_fallback_executor(executor);

            switch (fallback_executor) {
                case CPU:
                    throw sokudo::errors::ResolutionException("CPU implementation not found");
                case OPENCL:
#ifdef SOKUDO_OPENCL
                    return dynamic_cast<Task *>(sokudo::opencl::kernels::blas::cl_sasum(a, incx, res));
#else
                    throw sokudo::errors::ResolutionException("OpenCL implementation not found");
#endif
                case CUDA:
#ifdef SOKUDO_CUDA
                    return dynamic_cast<Task *>(cuda_wrapper::asum::cuda_sasum(a, incx, res));
#else
                    throw sokudo::errors::ResolutionException("CUDA implementation not found");
#endif
                default:
                    throw sokudo::errors::InvalidOperationException("Undefined task executor");
            }
        }

        // dasum
        Task *operator()(
                const sokudo::Buffer<double> &a,
                const sokudo::Value<uint64_t> &incx,
                const sokudo::Value<double> &res
        ) const {
            TaskExecutor fallback_executor = get_fallback_executor(executor);

            switch (fallback_executor) {
                case CPU:
                    throw sokudo::errors::ResolutionException("CPU implementation not found");
                case OPENCL:
#ifdef SOKUDO_OPENCL
                    return dynamic_cast<Task *>(sokudo::opencl::kernels::blas::cl_dasum(a, incx, res));
#else
                    throw sokudo::errors::ResolutionException("OpenCL implementation not found");
#endif
                case CUDA:
#ifdef SOKUDO_CUDA
                    return dynamic_cast<Task *>(cuda_wrapper::asum::cuda_dasum(a, incx, res));
#else
                    throw sokudo::errors::ResolutionException("CUDA implementation not found");
#endif
                default:
                    throw sokudo::errors::InvalidOperationException("Undefined task executor");
            }
        }

        //scasum
        Task *operator()(
                const sokudo::Buffer<float2> &a,
                const sokudo::Value<uint64_t> &incx,
                const sokudo::Value<float> &res
        ) const {
            TaskExecutor fallback_executor = get_fallback_executor(executor);

            switch (fallback_executor) {
                case CPU:
                    throw sokudo::errors::ResolutionException("CPU implementation not found");
                case OPENCL:
#ifdef SOKUDO_OPENCL
                    return dynamic_cast<Task *>(sokudo::opencl::kernels::blas::cl_scasum(a, incx, res));
#else
                    throw sokudo::errors::ResolutionException("OpenCL implementation not found");
#endif
                case CUDA:
#ifdef SOKUDO_CUDA
                    return dynamic_cast<Task *>(cuda_wrapper::asum::cuda_scasum(a, incx, res));
#else
                    throw sokudo::errors::ResolutionException("CUDA implementation not found");
#endif
                default:
                    throw sokudo::errors::InvalidOperationException("Undefined task executor");
            }
        }

        //dcasum
        Task *operator()(
                const sokudo::Buffer<double2> &a,
                const sokudo::Value<uint64_t> &incx,
                const sokudo::Value<double> &res
        ) const {
            TaskExecutor fallback_executor = get_fallback_executor(executor);

            switch (fallback_executor) {
                case CPU:
                    throw sokudo::errors::ResolutionException("CPU implementation not found");
                case OPENCL:
#ifdef SOKUDO_OPENCL
                    return dynamic_cast<Task *>(sokudo::opencl::kernels::blas::cl_dcasum(a, incx, res));
#else
                    throw sokudo::errors::ResolutionException("OpenCL implementation not found");
#endif
                case CUDA:
#ifdef SOKUDO_CUDA
                    return dynamic_cast<Task *>(cuda_wrapper::asum::cuda_dcasum(a, incx, res));
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
