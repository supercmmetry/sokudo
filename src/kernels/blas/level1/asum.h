#ifndef SOKUDO_BLAS_LEVEL1_ASUM_H
#define SOKUDO_BLAS_LEVEL1_ASUM_H

#include <task.h>
#include <types.h>

#ifdef SOKUDO_CUDA

#include <sokudo_cuda/blas/level1/asum.h>

#endif

#ifdef SOKUDO_OPENCL

#include <sokudo_opencl/blas/level1/asum.h>
#include <common.h>

#endif

namespace sokudo::kernels::blas {
#ifdef SOKUDO_CUDA
    namespace cuda_wrapper::asum {
        CUDATask *cuda_sasum(const sokudo::Value<uint64_t> &n, const sokudo::Buffer<float> &x,
                             const sokudo::Value<uint64_t> &incx,
                             const sokudo::Value<float> &res);

        CUDATask *cuda_dasum(const sokudo::Value<uint64_t> &n, const sokudo::Buffer<double> &x,
                             const sokudo::Value<uint64_t> &incx,
                             const sokudo::Value<double> &res);

        CUDATask *cuda_scasum(const sokudo::Value<uint64_t> &n, const sokudo::Buffer<float2> &x,
                              const sokudo::Value<uint64_t> &incx,
                              const sokudo::Value<float> &res);

        CUDATask *cuda_dcasum(const sokudo::Value<uint64_t> &n, const sokudo::Buffer<double2> &x,
                              const sokudo::Value<uint64_t> &incx,
                              const sokudo::Value<double> &res);
    }
#endif

    template<TaskExecutor executor>
    class Asum {
    public:
        Asum() = default;

        // sasum
        Task *operator()(
                const sokudo::Value<uint64_t> &n,
                const sokudo::Buffer<float> &x,
                const sokudo::Value<uint64_t> &incx,
                const sokudo::Value<float> &res
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
                    return dynamic_cast<Task *>(sokudo::opencl::kernels::blas::cl_sasum(n, x, incx, res));
#else
                    throw sokudo::errors::ResolutionException("OpenCL implementation not found");
#endif
                case CUDA:
#ifdef SOKUDO_CUDA
                    return dynamic_cast<Task *>(cuda_wrapper::asum::cuda_sasum(n, x, incx, res));
#else
                    throw sokudo::errors::ResolutionException("CUDA implementation not found");
#endif
                default:
                    throw sokudo::errors::InvalidOperationException("Undefined task executor");
            }
        }

        // dasum
        Task *operator()(
                const sokudo::Value<uint64_t> &n,
                const sokudo::Buffer<double> &x,
                const sokudo::Value<uint64_t> &incx,
                const sokudo::Value<double> &res
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
                    return dynamic_cast<Task *>(sokudo::opencl::kernels::blas::cl_dasum(n, x, incx, res));
#else
                    throw sokudo::errors::ResolutionException("OpenCL implementation not found");
#endif
                case CUDA:
#ifdef SOKUDO_CUDA
                    return dynamic_cast<Task *>(cuda_wrapper::asum::cuda_dasum(n, x, incx, res));
#else
                    throw sokudo::errors::ResolutionException("CUDA implementation not found");
#endif
                default:
                    throw sokudo::errors::InvalidOperationException("Undefined task executor");
            }
        }

        //scasum
        Task *operator()(
                const sokudo::Value<uint64_t> &n,
                const sokudo::Buffer<float2> &x,
                const sokudo::Value<uint64_t> &incx,
                const sokudo::Value<float> &res
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
                    return dynamic_cast<Task *>(sokudo::opencl::kernels::blas::cl_scasum(n, x, incx, res));
#else
                    throw sokudo::errors::ResolutionException("OpenCL implementation not found");
#endif
                case CUDA:
#ifdef SOKUDO_CUDA
                    return dynamic_cast<Task *>(cuda_wrapper::asum::cuda_scasum(n, x, incx, res));
#else
                    throw sokudo::errors::ResolutionException("CUDA implementation not found");
#endif
                default:
                    throw sokudo::errors::InvalidOperationException("Undefined task executor");
            }
        }

        //dcasum
        Task *operator()(
                const sokudo::Value<uint64_t> &n,
                const sokudo::Buffer<double2> &x,
                const sokudo::Value<uint64_t> &incx,
                const sokudo::Value<double> &res
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
                    return dynamic_cast<Task *>(sokudo::opencl::kernels::blas::cl_dcasum(n, x, incx, res));
#else
                    throw sokudo::errors::ResolutionException("OpenCL implementation not found");
#endif
                case CUDA:
#ifdef SOKUDO_CUDA
                    return dynamic_cast<Task *>(cuda_wrapper::asum::cuda_dcasum(n, x, incx, res));
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
