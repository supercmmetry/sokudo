#include <kernel_enums.h>
#include <sokudo_opencl/cl_helper.h>
#include <types.h>
#include "asum.h"

#ifndef SOKUDO_OPENCL_BLAS_SASUM_WGS
#define SOKUDO_OPENCL_BLAS_SASUM_WGS kernel.getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(device)
#endif

#ifndef SOKUDO_OPENCL_BLAS_SASUM_STRIDE
#define SOKUDO_OPENCL_BLAS_SASUM_STRIDE 0x10
#endif

#ifndef SOKUDO_OPENCL_BLAS_DASUM_WGS
#define SOKUDO_OPENCL_BLAS_DASUM_WGS kernel.getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(device)
#endif

#ifndef SOKUDO_OPENCL_BLAS_DASUM_STRIDE
#define SOKUDO_OPENCL_BLAS_DASUM_STRIDE 0x10
#endif

#ifndef SOKUDO_OPENCL_BLAS_SCASUM_WGS
#define SOKUDO_OPENCL_BLAS_SCASUM_WGS kernel.getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(device)
#endif

#ifndef SOKUDO_OPENCL_BLAS_SCASUM_STRIDE
#define SOKUDO_OPENCL_BLAS_SCASUM_STRIDE 0x10
#endif

#ifndef SOKUDO_OPENCL_BLAS_DCASUM_WGS
#define SOKUDO_OPENCL_BLAS_DCASUM_WGS kernel.getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(device)
#endif

#ifndef SOKUDO_OPENCL_BLAS_DCASUM_STRIDE
#define SOKUDO_OPENCL_BLAS_DCASUM_STRIDE 0x10
#endif


void sokudo::opencl::kernels::blas::register_sasum() {
    sokudo::opencl::ProgramProvider::register_kernel(sokudo::KERNEL_BLAS_SASUM,

#include "sasum.cl"

    );
}

void sokudo::opencl::kernels::blas::register_dasum() {
    sokudo::opencl::ProgramProvider::register_kernel(sokudo::KERNEL_BLAS_DASUM,

#include "dasum.cl"

    );
}

void sokudo::opencl::kernels::blas::register_scasum() {
    sokudo::opencl::ProgramProvider::register_kernel(sokudo::KERNEL_BLAS_SCASUM,

#include "scasum.cl"

    );
}


sokudo::CLTask *sokudo::opencl::kernels::blas::cl_sasum(
        const sokudo::DataBuffer<float> &x,
        const sokudo::DataValue<uint64_t> &incx,
        const sokudo::DataValue<float> &res,
        uint64_t wgs,
        uint64_t stride
) {
    // register kernel
    register_sasum();

    auto kernel = KernelProvider::get(sokudo::KERNEL_BLAS_SASUM);
    auto context = kernel.getInfo<CL_KERNEL_CONTEXT>();
    auto device = context.getInfo<CL_CONTEXT_DEVICES>().front();

    uint64_t n = x.size();

    auto queue = cl::CommandQueue(context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);

    cl::Buffer buf_a(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR, x.bsize(), x.inner());
    uint64_t m = 1;
    uint64_t s = stride == 0 ? SOKUDO_OPENCL_BLAS_SASUM_STRIDE : stride;
    auto local_size = wgs == 0 ? SOKUDO_OPENCL_BLAS_SASUM_WGS : wgs;
    auto global_size = (n / s + (n % s != 0));
    global_size = (global_size / incx.value() + (global_size % incx.value() != 0));
    global_size = (global_size / local_size + (global_size % local_size != 0)) * local_size;

    while (m < n) {
        kernel.setArg(0, buf_a);
        kernel.setArg(1, s);
        kernel.setArg(2, n);
        kernel.setArg(3, m);
        kernel.setArg(4, incx.value());
        m *= s;

        queue.enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(global_size), cl::NDRange(local_size));
        queue.enqueueBarrierWithWaitList();

        global_size = (global_size / s + (global_size % s != 0));
        global_size = (global_size / local_size + (global_size % local_size != 0)) * local_size;
    }

    queue.enqueueReadBuffer(buf_a, CL_FALSE, 0, res.bsize(), res.inner());
    return new CLTask(queue);
}

sokudo::CLTask *sokudo::opencl::kernels::blas::cl_dasum(
        const sokudo::DataBuffer<double> &x,
        const sokudo::DataValue<uint64_t> &incx,
        const sokudo::DataValue<double> &res,
        uint64_t wgs,
        uint64_t stride
) {
    // register kernel
    register_dasum();

    auto kernel = KernelProvider::get(sokudo::KERNEL_BLAS_DASUM);
    auto context = kernel.getInfo<CL_KERNEL_CONTEXT>();
    auto device = context.getInfo<CL_CONTEXT_DEVICES>().front();

    uint64_t n = x.size();

    auto queue = cl::CommandQueue(context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);

    cl::Buffer buf_a(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR, x.bsize(), x.inner());
    uint64_t m = 1;
    uint64_t s = stride == 0 ? SOKUDO_OPENCL_BLAS_DASUM_STRIDE : stride;
    auto local_size = wgs == 0 ? SOKUDO_OPENCL_BLAS_DASUM_WGS : wgs;
    auto global_size = (n / s + (n % s != 0));
    global_size = (global_size / incx.value() + (global_size % incx.value() != 0));
    global_size = (global_size / local_size + (global_size % local_size != 0)) * local_size;

    while (m < n) {
        kernel.setArg(0, buf_a);
        kernel.setArg(1, s);
        kernel.setArg(2, n);
        kernel.setArg(3, m);
        kernel.setArg(4, incx.value());
        m *= s;

        queue.enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(global_size), cl::NDRange(local_size));
        queue.enqueueBarrierWithWaitList();

        global_size = (global_size / s + (global_size % s != 0));
        global_size = (global_size / local_size + (global_size % local_size != 0)) * local_size;
    }

    queue.enqueueReadBuffer(buf_a, CL_FALSE, 0, res.bsize(), res.inner());
    return new CLTask(queue);
}

sokudo::CLTask *sokudo::opencl::kernels::blas::cl_scasum(
        const sokudo::DataBuffer<float2> &x,
        const sokudo::DataValue<uint64_t> &incx,
        const sokudo::DataValue<float> &res,
        uint64_t wgs,
        uint64_t stride
) {
    // register kernel
    register_scasum();

    auto kernel = KernelProvider::get(sokudo::KERNEL_BLAS_SCASUM);
    auto context = kernel.getInfo<CL_KERNEL_CONTEXT>();
    auto device = context.getInfo<CL_CONTEXT_DEVICES>().front();

    uint64_t n = x.size();

    auto queue = cl::CommandQueue(context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);
    cl::Buffer buf_a(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR, x.bsize(), x.inner());
    uint64_t m = 1;
    uint64_t s = stride == 0 ? SOKUDO_OPENCL_BLAS_SCASUM_STRIDE : stride;
    auto local_size = wgs == 0 ? SOKUDO_OPENCL_BLAS_SCASUM_WGS : wgs;
    auto global_size = (n / s + (n % s != 0));
    global_size = (global_size / incx.value() + (global_size % incx.value() != 0));
    global_size = (global_size / local_size + (global_size % local_size != 0)) * local_size;

    while (m < n) {
        kernel.setArg(0, buf_a);
        kernel.setArg(1, s);
        kernel.setArg(2, n);
        kernel.setArg(3, m);
        kernel.setArg(4, incx.value());
        m *= s;

        queue.enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(global_size), cl::NDRange(local_size));
        queue.enqueueBarrierWithWaitList();

        global_size = (global_size / s + (global_size % s != 0));
        global_size = (global_size / local_size + (global_size % local_size != 0)) * local_size;
    }

    queue.enqueueReadBuffer(buf_a, CL_FALSE, 0, res.bsize(), res.inner());
    return new CLTask(queue);
}