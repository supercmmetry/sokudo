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
        CUDATask *cuda_samax(const sokudo::Value<uint64_t> &n, const sokudo::Buffer<float> &x,
                             const sokudo::Value<uint64_t> &incx,
                             const sokudo::Value<uint64_t> &res);

        CUDATask *cuda_damax(const sokudo::Value<uint64_t> &n, const sokudo::Buffer<double> &x,
                             const sokudo::Value<uint64_t> &incx,
                             const sokudo::Value<uint64_t> &res);

        CUDATask *cuda_scamax(const sokudo::Value<uint64_t> &n, const sokudo::Buffer<float2> &x,
                              const sokudo::Value<uint64_t> &incx,
                              const sokudo::Value<uint64_t> &res);

        CUDATask *cuda_dcamax(const sokudo::Value<uint64_t> &n, const sokudo::Buffer<double2> &x,
                              const sokudo::Value<uint64_t> &incx,
                              const sokudo::Value<uint64_t> &res);
    }
#endif

    template<TaskExecutor executor>
    class Amax {
    public:
        Amax() = default;

        // samax
        Task *operator()(
                const sokudo::Value<uint64_t> &n,
                const sokudo::Buffer<float> &x,
                const sokudo::Value<uint64_t> &incx,
                const sokudo::Value<uint64_t> &res
        ) const {
            sokudo_assert(n.value() > 0, "Vector cannot be empty");
            sokudo_assert(incx.value() > 0, "Stride cannot be zero");
            sokudo_assert(1 + (n.value() - 1) * incx.value() <= x.size(), "Index out of bounds");
            TaskExecutor fallback_executor = get_fallback_executor(executor);

            switch (fallback_executor) {
                case CPU:
                    throw sokudo::errors::ResolutionException("CPU implementation not found");
                case OPENCL:
#ifdef SOKUDO_OPENCL
                    return dynamic_cast<Task *>(sokudo::opencl::kernels::blas::cl_samax(n, x, incx, res));
#else
                    throw sokudo::errors::ResolutionException("OpenCL implementation not found");
#endif
                case CUDA:
#ifdef SOKUDO_CUDA
                    return dynamic_cast<Task *>(cuda_wrapper::amax::cuda_samax(n, x, incx, res));
#else
                    throw sokudo::errors::ResolutionException("CUDA implementation not found");
#endif
                default:
                    throw sokudo::errors::InvalidOperationException("Undefined task executor");
            }
        }

        // damax
        Task *operator()(
                const sokudo::Value<uint64_t> &n,
                const sokudo::Buffer<double> &x,
                const sokudo::Value<uint64_t> &incx,
                const sokudo::Value<uint64_t> &res
        ) const {
            sokudo_assert(n.value() > 0, "Vector cannot be empty");
            sokudo_assert(incx.value() > 0, "Stride cannot be zero");
            sokudo_assert(1 + (n.value() - 1) * incx.value() <= x.size(), "Index out of bounds");
            TaskExecutor fallback_executor = get_fallback_executor(executor);

            switch (fallback_executor) {
                case CPU:
                    throw sokudo::errors::ResolutionException("CPU implementation not found");
                case OPENCL:
#ifdef SOKUDO_OPENCL
                    return dynamic_cast<Task *>(sokudo::opencl::kernels::blas::cl_damax(n, x, incx, res));
#else
                    throw sokudo::errors::ResolutionException("OpenCL implementation not found");
#endif
                case CUDA:
#ifdef SOKUDO_CUDA
                    return dynamic_cast<Task *>(cuda_wrapper::amax::cuda_damax(n, x, incx, res));
#else
                    throw sokudo::errors::ResolutionException("CUDA implementation not found");
#endif
                default:
                    throw sokudo::errors::InvalidOperationException("Undefined task executor");
            }
        }

        //scamax
        Task *operator()(
                const sokudo::Value<uint64_t> &n,
                const sokudo::Buffer<float2> &x,
                const sokudo::Value<uint64_t> &incx,
                const sokudo::Value<uint64_t> &res
        ) const {
            sokudo_assert(n.value() > 0, "Vector cannot be empty");
            sokudo_assert(incx.value() > 0, "Stride cannot be zero");
            sokudo_assert(1 + (n.value() - 1) * incx.value() <= x.size(), "Index out of bounds");
            TaskExecutor fallback_executor = get_fallback_executor(executor);

            switch (fallback_executor) {
                case CPU:
                    throw sokudo::errors::ResolutionException("CPU implementation not found");
                case OPENCL:
#ifdef SOKUDO_OPENCL
                    return dynamic_cast<Task *>(sokudo::opencl::kernels::blas::cl_scamax(n, x, incx, res));
#else
                    throw sokudo::errors::ResolutionException("OpenCL implementation not found");
#endif
                case CUDA:
#ifdef SOKUDO_CUDA
                    return dynamic_cast<Task *>(cuda_wrapper::amax::cuda_scamax(n, x, incx, res));
#else
                    throw sokudo::errors::ResolutionException("CUDA implementation not found");
#endif
                default:
                    throw sokudo::errors::InvalidOperationException("Undefined task executor");
            }
        }

        //dcamax
        Task *operator()(
                const sokudo::Value<uint64_t> &n,
                const sokudo::Buffer<double2> &x,
                const sokudo::Value<uint64_t> &incx,
                const sokudo::Value<uint64_t> &res
        ) const {
            sokudo_assert(n.value() > 0, "Vector cannot be empty");
            sokudo_assert(incx.value() > 0, "Stride cannot be zero");
            sokudo_assert(1 + (n.value() - 1) * incx.value() <= x.size(), "Index out of bounds");
            TaskExecutor fallback_executor = get_fallback_executor(executor);

            switch (fallback_executor) {
                case CPU:
                    throw sokudo::errors::ResolutionException("CPU implementation not found");
                case OPENCL:
#ifdef SOKUDO_OPENCL
                    return dynamic_cast<Task *>(sokudo::opencl::kernels::blas::cl_dcamax(n, x, incx, res));
#else
                    throw sokudo::errors::ResolutionException("OpenCL implementation not found");
#endif
                case CUDA:
#ifdef SOKUDO_CUDA
                    return dynamic_cast<Task *>(cuda_wrapper::amax::cuda_dcamax(n, x, incx, res));
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
