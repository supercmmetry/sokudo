#ifndef SOKUDO_BLAS_AXPY_H
#define SOKUDO_BLAS_AXPY_H

#include <task.h>
#include <types.h>
#include <common.h>


#ifdef SOKUDO_OPENCL

#include <sokudo_opencl/blas/level1/axpy.h>

#endif

namespace sokudo::kernels::blas {
#ifdef SOKUDO_CUDA
    namespace cuda_wrapper::axpy {
        CUDATask *cuda_saxpy(
                const sokudo::Value<uint64_t> &n,
                const sokudo::Value<float> &alpha,
                const sokudo::Buffer<float> &x,
                const sokudo::Value<uint64_t> &incx,
                const sokudo::Buffer<float> &y,
                const sokudo::Value<uint64_t> &incy
        );

        CUDATask *cuda_daxpy(
                const sokudo::Value<uint64_t> &n,
                const sokudo::Value<double> &alpha,
                const sokudo::Buffer<double> &x,
                const sokudo::Value<uint64_t> &incx,
                const sokudo::Buffer<double> &y,
                const sokudo::Value<uint64_t> &incy
        );

        CUDATask *cuda_scaxpy(
                const sokudo::Value<uint64_t> &n,
                const sokudo::Value<float2> &alpha,
                const sokudo::Buffer<float2> &x,
                const sokudo::Value<uint64_t> &incx,
                const sokudo::Buffer<float2> &y,
                const sokudo::Value<uint64_t> &incy
        );

        CUDATask *cuda_dcaxpy(
                const sokudo::Value<uint64_t> &n,
                const sokudo::Value<double2> &alpha,
                const sokudo::Buffer<double2> &x,
                const sokudo::Value<uint64_t> &incx,
                const sokudo::Buffer<double2> &y,
                const sokudo::Value<uint64_t> &incy
        );
    }
#endif

    template<TaskExecutor executor>
    class Axpy {
    public:
        Axpy() = default;

        // saxpy
        Task *operator()(
                const sokudo::Value<uint64_t> &n,
                const sokudo::Value<float> &alpha,
                const sokudo::Buffer<float> &x,
                const sokudo::Value<uint64_t> &incx,
                const sokudo::Buffer<float> &y,
                const sokudo::Value<uint64_t> &incy
        ) const {
            sokudo_assert(n.value() > 0, "Vector cannot be empty");
            sokudo_assert(incx.value() > 0, "Stride cannot be zero");
            sokudo_assert(incy.value() > 0, "Stride cannot be zero");
            sokudo_assert(1 + (n.value() - 1) * incx.value() <= x.size(), "Index out of bounds");
            sokudo_assert(1 + (n.value() - 1) * incy.value() <= y.size(), "Index out of bounds");
            TaskExecutor fallback_executor = get_fallback_executor(executor);

            switch (fallback_executor) {
                case CPU:
                    throw sokudo::errors::ResolutionException("CPU implementation not found");
                case OPENCL:
#ifdef SOKUDO_OPENCL
                    return dynamic_cast<Task *>(sokudo::opencl::kernels::blas::cl_saxpy(n, alpha, x, incx, y, incy));
#else
                    throw sokudo::errors::ResolutionException("OpenCL implementation not found");
#endif
                case CUDA:
#ifdef SOKUDO_CUDA
                    return dynamic_cast<Task *>(cuda_wrapper::axpy::cuda_saxpy(n, alpha, x, incx, y, incy));
#else
                    throw sokudo::errors::ResolutionException("CUDA implementation not found");
#endif
                default:
                    throw sokudo::errors::InvalidOperationException("Undefined task executor");
            }
        }

        // daxpy
        Task *operator()(
                const sokudo::Value<uint64_t> &n,
                const sokudo::Value<double> &alpha,
                const sokudo::Buffer<double> &x,
                const sokudo::Value<uint64_t> &incx,
                const sokudo::Buffer<double> &y,
                const sokudo::Value<uint64_t> &incy
        ) const {
            sokudo_assert(n.value() > 0, "Vector cannot be empty");
            sokudo_assert(incx.value() > 0, "Stride cannot be zero");
            sokudo_assert(incy.value() > 0, "Stride cannot be zero");
            sokudo_assert(1 + (n.value() - 1) * incx.value() <= x.size(), "Index out of bounds");
            sokudo_assert(1 + (n.value() - 1) * incy.value() <= y.size(), "Index out of bounds");
            TaskExecutor fallback_executor = get_fallback_executor(executor);

            switch (fallback_executor) {
                case CPU:
                    throw sokudo::errors::ResolutionException("CPU implementation not found");
                case OPENCL:
#ifdef SOKUDO_OPENCL
                    return dynamic_cast<Task *>(sokudo::opencl::kernels::blas::cl_daxpy(n, alpha, x, incx, y, incy));
#else
                    throw sokudo::errors::ResolutionException("OpenCL implementation not found");
#endif
                case CUDA:
#ifdef SOKUDO_CUDA
                    return dynamic_cast<Task *>(cuda_wrapper::axpy::cuda_daxpy(n, alpha, x, incx, y, incy));
#else
                    throw sokudo::errors::ResolutionException("CUDA implementation not found");
#endif
                default:
                    throw sokudo::errors::InvalidOperationException("Undefined task executor");
            }
        }

        //scaxpy
        Task *operator()(
                const sokudo::Value<uint64_t> &n,
                const sokudo::Value<float2> &alpha,
                const sokudo::Buffer<float2> &x,
                const sokudo::Value<uint64_t> &incx,
                const sokudo::Buffer<float2> &y,
                const sokudo::Value<uint64_t> &incy
        ) const {
            sokudo_assert(n.value() > 0, "Vector cannot be empty");
            sokudo_assert(incx.value() > 0, "Stride cannot be zero");
            sokudo_assert(incy.value() > 0, "Stride cannot be zero");
            sokudo_assert(1 + (n.value() - 1) * incx.value() <= x.size(), "Index out of bounds");
            sokudo_assert(1 + (n.value() - 1) * incy.value() <= y.size(), "Index out of bounds");
            TaskExecutor fallback_executor = get_fallback_executor(executor);

            switch (fallback_executor) {
                case CPU:
                    throw sokudo::errors::ResolutionException("CPU implementation not found");
                case OPENCL:
#ifdef SOKUDO_OPENCL
                    return dynamic_cast<Task *>(sokudo::opencl::kernels::blas::cl_scaxpy(n, alpha, x, incx, y, incy));
#else
                    throw sokudo::errors::ResolutionException("OpenCL implementation not found");
#endif
                case CUDA:
#ifdef SOKUDO_CUDA
                    return dynamic_cast<Task *>(cuda_wrapper::axpy::cuda_scaxpy(n, alpha, x, incx, y, incy));
#else
                    throw sokudo::errors::ResolutionException("CUDA implementation not found");
#endif
                default:
                    throw sokudo::errors::InvalidOperationException("Undefined task executor");
            }
        }

        //dcaxpy
        Task *operator()(
                const sokudo::Value<uint64_t> &n,
                const sokudo::Value<double2> &alpha,
                const sokudo::Buffer<double2> &x,
                const sokudo::Value<uint64_t> &incx,
                const sokudo::Buffer<double2> &y,
                const sokudo::Value<uint64_t> &incy
        ) const {
            sokudo_assert(n.value() > 0, "Vector cannot be empty");
            sokudo_assert(incx.value() > 0, "Stride cannot be zero");
            sokudo_assert(incy.value() > 0, "Stride cannot be zero");
            sokudo_assert(1 + (n.value() - 1) * incx.value() <= x.size(), "Index out of bounds");
            sokudo_assert(1 + (n.value() - 1) * incy.value() <= y.size(), "Index out of bounds");
            TaskExecutor fallback_executor = get_fallback_executor(executor);

            switch (fallback_executor) {
                case CPU:
                    throw sokudo::errors::ResolutionException("CPU implementation not found");
                case OPENCL:
#ifdef SOKUDO_OPENCL
                    return dynamic_cast<Task *>(sokudo::opencl::kernels::blas::cl_dcaxpy(n, alpha, x, incx, y, incy));
#else
                    throw sokudo::errors::ResolutionException("OpenCL implementation not found");
#endif
                case CUDA:
#ifdef SOKUDO_CUDA
                    return dynamic_cast<Task *>(cuda_wrapper::axpy::cuda_dcaxpy(n, alpha, x, incx, y, incy));
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
