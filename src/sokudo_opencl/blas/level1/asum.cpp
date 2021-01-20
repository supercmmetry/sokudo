#include <kernel_enums.h>
#include <sokudo_opencl/cl_helper.h>
#include <types.h>
#include <common.h>
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


void sokudo::opencl::kernels::blas::register_asum() {
    sokudo::opencl::ProgramProvider::register_kernel(sokudo::KERNEL_BLAS_ASUM,

#include "asum.cl"

    );
}

void detail_asum_cl(
        sokudo::CLTask *task,
        const std::string &name,
        const std::string &n,
        const std::string &x,
        const std::string &incx,
        const std::string &res,
        uint64_t wgs,
        uint64_t stride
) {

    task->set(sokudo::KERNEL_BLAS_ASUM, name);
    task->params()["n"] = n;
    task->params()["x"] = x;
    task->params()["incx"] = incx;
    task->params()["res"] = res;
    task->params()["wgs"] = std::to_string(wgs);
    task->params()["stride"] = std::to_string(stride);
}

sokudo::CLTask *sokudo::opencl::kernels::blas::cl_sasum(
        const sokudo::Value<uint64_t> &_n,
        const sokudo::Buffer<float> &x,
        const sokudo::Value<uint64_t> &incx,
        const sokudo::Value<float> &res,
        uint64_t wgs,
        uint64_t stride
) {
    // register kernel
    register_asum();

    auto kernel = KernelProvider::get(sokudo::KERNEL_BLAS_ASUM, "sasum");
    auto context = kernel.getInfo<CL_KERNEL_CONTEXT>();
    auto device = context.getInfo<CL_CONTEXT_DEVICES>().front();

    uint64_t n = _n.value();

    auto queue = cl::CommandQueue(context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);

    cl::Buffer buf_a(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR, x.bsize(), x.inner());
    uint64_t m = 1;
    uint64_t s = stride == 0 ? SOKUDO_OPENCL_BLAS_SASUM_STRIDE : stride;
    auto local_size = wgs == 0 ? SOKUDO_OPENCL_BLAS_SASUM_WGS : wgs;
    auto global_size = (n / s + (n % s != 0));
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

    auto task = new CLTask(queue);
    detail_asum_cl(task, "BLAS_SASUM", _n.name(), x.name(), incx.name(), res.name(), local_size, s);
    return task;
}

sokudo::CLTask *sokudo::opencl::kernels::blas::cl_dasum(
        const sokudo::Value<uint64_t> &_n,
        const sokudo::Buffer<double> &x,
        const sokudo::Value<uint64_t> &incx,
        const sokudo::Value<double> &res,
        uint64_t wgs,
        uint64_t stride
) {
    // register kernel
    register_asum();

    auto kernel = KernelProvider::get(sokudo::KERNEL_BLAS_ASUM, "dasum");
    auto context = kernel.getInfo<CL_KERNEL_CONTEXT>();
    auto device = context.getInfo<CL_CONTEXT_DEVICES>().front();

    uint64_t n = _n.value();

    auto queue = cl::CommandQueue(context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);

    cl::Buffer buf_a(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR, x.bsize(), x.inner());
    uint64_t m = 1;
    uint64_t s = stride == 0 ? SOKUDO_OPENCL_BLAS_DASUM_STRIDE : stride;
    auto local_size = wgs == 0 ? SOKUDO_OPENCL_BLAS_DASUM_WGS : wgs;
    auto global_size = (n / s + (n % s != 0));
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

    auto task = new CLTask(queue);
    detail_asum_cl(task, "BLAS_DASUM", _n.name(), x.name(), incx.name(), res.name(), local_size, s);
    return task;
}

sokudo::CLTask *sokudo::opencl::kernels::blas::cl_scasum(
        const sokudo::Value<uint64_t> &_n,
        const sokudo::Buffer<float2> &x,
        const sokudo::Value<uint64_t> &incx,
        const sokudo::Value<float> &res,
        uint64_t wgs,
        uint64_t stride
) {
    // register kernel
    register_asum();

    auto kernel = KernelProvider::get(sokudo::KERNEL_BLAS_ASUM, "scasum");
    auto context = kernel.getInfo<CL_KERNEL_CONTEXT>();
    auto device = context.getInfo<CL_CONTEXT_DEVICES>().front();

    uint64_t n = _n.value();

    auto queue = cl::CommandQueue(context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);
    cl::Buffer buf_a(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR, x.bsize(), x.inner());
    uint64_t m = 1;
    uint64_t s = stride == 0 ? SOKUDO_OPENCL_BLAS_SCASUM_STRIDE : stride;
    auto local_size = wgs == 0 ? SOKUDO_OPENCL_BLAS_SCASUM_WGS : wgs;
    auto global_size = (n / s + (n % s != 0));
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
    auto task = new CLTask(queue);
    detail_asum_cl(task, "BLAS_SCASUM", _n.name(), x.name(), incx.name(), res.name(), local_size, s);
    return task;
}

sokudo::CLTask *sokudo::opencl::kernels::blas::cl_dcasum(
        const sokudo::Value<uint64_t> &_n,
        const sokudo::Buffer<double2> &x,
        const sokudo::Value<uint64_t> &incx,
        const sokudo::Value<double> &res,
        uint64_t wgs,
        uint64_t stride
) {
    // register kernel
    register_asum();

    auto kernel = KernelProvider::get(sokudo::KERNEL_BLAS_ASUM, "dcasum");
    auto context = kernel.getInfo<CL_KERNEL_CONTEXT>();
    auto device = context.getInfo<CL_CONTEXT_DEVICES>().front();

    uint64_t n = _n.value();

    auto queue = cl::CommandQueue(context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);
    cl::Buffer buf_a(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR, x.bsize(), x.inner());
    uint64_t m = 1;
    uint64_t s = stride == 0 ? SOKUDO_OPENCL_BLAS_DCASUM_STRIDE : stride;
    auto local_size = wgs == 0 ? SOKUDO_OPENCL_BLAS_DCASUM_WGS : wgs;
    auto global_size = (n / s + (n % s != 0));
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

    auto task = new CLTask(queue);
    detail_asum_cl(task, "BLAS_DCASUM", _n.name(), x.name(), incx.name(), res.name(), local_size, s);
    return task;
}
