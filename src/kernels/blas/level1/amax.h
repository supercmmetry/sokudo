#ifndef SOKUDO_BLAS_LEVEL1_AMAX_H
#define SOKUDO_BLAS_LEVEL1_AMAX_H

#include <task.h>
#include <types.h>

#ifdef SOKUDO_CUDA

#include <sokudo_cuda/blas/level1/amax.h>

#endif

#ifdef SOKUDO_OPENCL

#include <sokudo_opencl/blas/level1/amax.h>

#endif

namespace sokudo::kernels::blas {
#ifdef SOKUDO_CUDA
    namespace cuda_wrapper::amax {
        CUDATask *cuda_samax(const sokudo::Buffer<float> &a, const sokudo::Value<uint64_t> &incx,
                             const sokudo::Value<uint64_t> &res);

        CUDATask *cuda_damax(const sokudo::Buffer<double> &a, const sokudo::Value<uint64_t> &incx,
                             const sokudo::Value<uint64_t> &res);

        CUDATask *cuda_scamax(const sokudo::Buffer<float2> &a, const sokudo::Value<uint64_t> &incx,
                              const sokudo::Value<uint64_t> &res);

        CUDATask *cuda_dcamax(const sokudo::Buffer<double2> &a, const sokudo::Value<uint64_t> &incx,
                              const sokudo::Value<uint64_t> &res);
    }
#endif

    template<TaskExecutor executor>
    class Amax {
    public:
        Amax() = default;

        // samax
        Task *operator()(
                const sokudo::Buffer<float> &a,
                const sokudo::Value<uint64_t> &incx,
                const sokudo::Value<uint64_t> &res
        ) const {
            TaskExecutor fallback_executor = get_fallback_executor(executor);

            switch (fallback_executor) {
                case CPU:
                    throw sokudo::errors::ResolutionException("CPU implementation not found");
                case OPENCL:
#ifdef SOKUDO_OPENCL
                    return dynamic_cast<Task *>(sokudo::opencl::kernels::blas::cl_samax(a, incx, res));
#else
                    throw sokudo::errors::ResolutionException("OpenCL implementation not found");
#endif
                case CUDA:
#ifdef SOKUDO_CUDA
                    return dynamic_cast<Task *>(cuda_wrapper::amax::cuda_samax(a, incx, res));
#else
                    throw sokudo::errors::ResolutionException("CUDA implementation not found");
#endif
                default:
                    throw sokudo::errors::InvalidOperationException("Undefined task executor");
            }
        }

        // damax
        Task *operator()(
                const sokudo::Buffer<double> &a,
                const sokudo::Value<uint64_t> &incx,
                const sokudo::Value<uint64_t> &res
        ) const {
            TaskExecutor fallback_executor = get_fallback_executor(executor);

            switch (fallback_executor) {
                case CPU:
                    throw sokudo::errors::ResolutionException("CPU implementation not found");
                case OPENCL:
#ifdef SOKUDO_OPENCL
                    return dynamic_cast<Task *>(sokudo::opencl::kernels::blas::cl_damax(a, incx, res));
#else
                    throw sokudo::errors::ResolutionException("OpenCL implementation not found");
#endif
                case CUDA:
#ifdef SOKUDO_CUDA
                    return dynamic_cast<Task *>(cuda_wrapper::amax::cuda_damax(a, incx, res));
#else
                    throw sokudo::errors::ResolutionException("CUDA implementation not found");
#endif
                default:
                    throw sokudo::errors::InvalidOperationException("Undefined task executor");
            }
        }

        //scamax
        Task *operator()(
                const sokudo::Buffer<float2> &a,
                const sokudo::Value<uint64_t> &incx,
                const sokudo::Value<uint64_t> &res
        ) const {
            TaskExecutor fallback_executor = get_fallback_executor(executor);

            switch (fallback_executor) {
                case CPU:
                    throw sokudo::errors::ResolutionException("CPU implementation not found");
                case OPENCL:
#ifdef SOKUDO_OPENCL
                    return dynamic_cast<Task *>(sokudo::opencl::kernels::blas::cl_scamax(a, incx, res));
#else
                    throw sokudo::errors::ResolutionException("OpenCL implementation not found");
#endif
                case CUDA:
#ifdef SOKUDO_CUDA
                    return dynamic_cast<Task *>(cuda_wrapper::amax::cuda_scamax(a, incx, res));
#else
                    throw sokudo::errors::ResolutionException("CUDA implementation not found");
#endif
                default:
                    throw sokudo::errors::InvalidOperationException("Undefined task executor");
            }
        }

        //dcamax
        Task *operator()(
                const sokudo::Buffer<double2> &a,
                const sokudo::Value<uint64_t> &incx,
                const sokudo::Value<uint64_t> &res
        ) const {
            TaskExecutor fallback_executor = get_fallback_executor(executor);

            switch (fallback_executor) {
                case CPU:
                    throw sokudo::errors::ResolutionException("CPU implementation not found");
                case OPENCL:
#ifdef SOKUDO_OPENCL
                    return dynamic_cast<Task *>(sokudo::opencl::kernels::blas::cl_dcamax(a, incx, res));
#else
                    throw sokudo::errors::ResolutionException("OpenCL implementation not found");
#endif
                case CUDA:
#ifdef SOKUDO_CUDA
                    return dynamic_cast<Task *>(cuda_wrapper::amax::cuda_dcamax(a, incx, res));
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
