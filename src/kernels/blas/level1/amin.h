#ifndef SOKUDO_BLAS_LEVEL1_AMIN_H
#define SOKUDO_BLAS_LEVEL1_AMIN_H

#include <task.h>
#include <types.h>

#ifdef SOKUDO_CUDA

#include <sokudo_cuda/blas/level1/amin.h>

#endif

#ifdef SOKUDO_OPENCL

#include <sokudo_opencl/blas/level1/amin.h>

#endif

namespace sokudo::kernels::blas {
#ifdef SOKUDO_CUDA
    namespace cuda_wrapper::amin {
        CUDATask *cuda_samin(const sokudo::Buffer<float> &a, const sokudo::Value<uint64_t> &incx,
                             const sokudo::Value<uint64_t> &res);

        CUDATask *cuda_damin(const sokudo::Buffer<double> &a, const sokudo::Value<uint64_t> &incx,
                             const sokudo::Value<uint64_t> &res);

        CUDATask *cuda_scamin(const sokudo::Buffer<float2> &a, const sokudo::Value<uint64_t> &incx,
                              const sokudo::Value<uint64_t> &res);

        CUDATask *cuda_dcamin(const sokudo::Buffer<double2> &a, const sokudo::Value<uint64_t> &incx,
                              const sokudo::Value<uint64_t> &res);
    }
#endif

    template<TaskExecutor executor>
    class Amin {
    public:
        Amin() = default;

        // samin
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
                    return dynamic_cast<Task *>(sokudo::opencl::kernels::blas::cl_samin(a, incx, res));
#else
                    throw sokudo::errors::ResolutionException("OpenCL implementation not found");
#endif
                case CUDA:
#ifdef SOKUDO_CUDA
                    return dynamic_cast<Task *>(cuda_wrapper::amin::cuda_samin(a, incx, res));
#else
                    throw sokudo::errors::ResolutionException("CUDA implementation not found");
#endif
                default:
                    throw sokudo::errors::InvalidOperationException("Undefined task executor");
            }
        }

        // damin
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
                    return dynamic_cast<Task *>(sokudo::opencl::kernels::blas::cl_damin(a, incx, res));
#else
                    throw sokudo::errors::ResolutionException("OpenCL implementation not found");
#endif
                case CUDA:
#ifdef SOKUDO_CUDA
                    return dynamic_cast<Task *>(cuda_wrapper::amin::cuda_damin(a, incx, res));
#else
                    throw sokudo::errors::ResolutionException("CUDA implementation not found");
#endif
                default:
                    throw sokudo::errors::InvalidOperationException("Undefined task executor");
            }
        }

        //scamin
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
                    return dynamic_cast<Task *>(sokudo::opencl::kernels::blas::cl_scamin(a, incx, res));
#else
                    throw sokudo::errors::ResolutionException("OpenCL implementation not found");
#endif
                case CUDA:
#ifdef SOKUDO_CUDA
                    return dynamic_cast<Task *>(cuda_wrapper::amin::cuda_scamin(a, incx, res));
#else
                    throw sokudo::errors::ResolutionException("CUDA implementation not found");
#endif
                default:
                    throw sokudo::errors::InvalidOperationException("Undefined task executor");
            }
        }

        //dcamin
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
                    return dynamic_cast<Task *>(sokudo::opencl::kernels::blas::cl_dcamin(a, incx, res));
#else
                    throw sokudo::errors::ResolutionException("OpenCL implementation not found");
#endif
                case CUDA:
#ifdef SOKUDO_CUDA
                    return dynamic_cast<Task *>(cuda_wrapper::amin::cuda_dcamin(a, incx, res));
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
