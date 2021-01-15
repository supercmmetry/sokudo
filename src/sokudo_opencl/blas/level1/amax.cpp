#include <kernel_enums.h>
#include <sokudo_opencl/cl_helper.h>
#include <types.h>
#include "amax.h"

#ifndef SOKUDO_OPENCL_BLAS_SAMAX_WGS
#define SOKUDO_OPENCL_BLAS_SAMAX_WGS kernel.getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(device)
#endif

#ifndef SOKUDO_OPENCL_BLAS_SAMAX_STRIDE
#define SOKUDO_OPENCL_BLAS_SAMAX_STRIDE 0x10
#endif

#ifndef SOKUDO_OPENCL_BLAS_DAMAX_WGS
#define SOKUDO_OPENCL_BLAS_DAMAX_WGS kernel.getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(device)
#endif

#ifndef SOKUDO_OPENCL_BLAS_DAMAX_STRIDE
#define SOKUDO_OPENCL_BLAS_DAMAX_STRIDE 0x10
#endif

#ifndef SOKUDO_OPENCL_BLAS_SCAMAX_WGS
#define SOKUDO_OPENCL_BLAS_SCAMAX_WGS kernel.getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(device)
#endif

#ifndef SOKUDO_OPENCL_BLAS_SCAMAX_STRIDE
#define SOKUDO_OPENCL_BLAS_SCAMAX_STRIDE 0x10
#endif

#ifndef SOKUDO_OPENCL_BLAS_DCAMAX_WGS
#define SOKUDO_OPENCL_BLAS_DCAMAX_WGS kernel.getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(device)
#endif

#ifndef SOKUDO_OPENCL_BLAS_DCAMAX_STRIDE
#define SOKUDO_OPENCL_BLAS_DCAMAX_STRIDE 0x10
#endif

void sokudo::opencl::kernels::blas::register_amax() {
    sokudo::opencl::ProgramProvider::register_kernel(sokudo::KERNEL_BLAS_AMAX,

#include "amax.cl"

    );
}

void detail(
        sokudo::CLTask *task,
        const std::string &name,
        const std::string &x,
        const std::string &incx,
        const std::string &res,
        uint64_t wgs,
        uint64_t stride
) {

    task->set(sokudo::KERNEL_BLAS_AMAX, name);
    task->params()["x"] = x;
    task->params()["incx"] = incx;
    task->params()["res"] = res;
    task->params()["wgs"] = std::to_string(wgs);
    task->params()["stride"] = std::to_string(stride);
}

sokudo::CLTask *
sokudo::opencl::kernels::blas::cl_samax(const sokudo::Buffer<float> &x, const sokudo::Value<uint64_t> &incx,
                                        const sokudo::Value<float> &res, uint64_t wgs, uint64_t stride) {
    // register kernel
    register_amax();

    auto kernel = KernelProvider::get(sokudo::KERNEL_BLAS_AMAX, "samax");
    auto context = kernel.getInfo<CL_KERNEL_CONTEXT>();
    auto device = context.getInfo<CL_CONTEXT_DEVICES>().front();

    uint64_t n = x.size();

    auto queue = cl::CommandQueue(context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);

    cl::Buffer buf_a(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR, x.bsize(), x.inner());
    uint64_t m = 1;
    uint64_t s = stride == 0 ? SOKUDO_OPENCL_BLAS_SAMAX_STRIDE : stride;
    auto local_size = wgs == 0 ? SOKUDO_OPENCL_BLAS_SAMAX_WGS : wgs;
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

    auto task = new CLTask(queue);
    detail(task, "BLAS_SAMAX", x.name(), incx.name(), res.name(), local_size, s);
    return task;
}

sokudo::CLTask *
sokudo::opencl::kernels::blas::cl_damax(const sokudo::Buffer<double> &x, const sokudo::Value<uint64_t> &incx,
                                        const sokudo::Value<double> &res, uint64_t wgs, uint64_t stride) {
    // register kernel
    register_amax();

    auto kernel = KernelProvider::get(sokudo::KERNEL_BLAS_AMAX, "damax");
    auto context = kernel.getInfo<CL_KERNEL_CONTEXT>();
    auto device = context.getInfo<CL_CONTEXT_DEVICES>().front();

    uint64_t n = x.size();

    auto queue = cl::CommandQueue(context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);

    cl::Buffer buf_a(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR, x.bsize(), x.inner());
    uint64_t m = 1;
    uint64_t s = stride == 0 ? SOKUDO_OPENCL_BLAS_DAMAX_STRIDE : stride;
    auto local_size = wgs == 0 ? SOKUDO_OPENCL_BLAS_DAMAX_WGS : wgs;
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

    auto task = new CLTask(queue);
    detail(task, "BLAS_DAMAX", x.name(), incx.name(), res.name(), local_size, s);
    return task;
}

sokudo::CLTask *
sokudo::opencl::kernels::blas::cl_scamax(const sokudo::Buffer<float2> &x, const sokudo::Value<uint64_t> &incx,
                                         const sokudo::Value<float> &res, uint64_t wgs, uint64_t stride) {
    // register kernel
    register_amax();

    auto kernel = KernelProvider::get(sokudo::KERNEL_BLAS_AMAX, "scamax");
    auto context = kernel.getInfo<CL_KERNEL_CONTEXT>();
    auto device = context.getInfo<CL_CONTEXT_DEVICES>().front();

    uint64_t n = x.size();

    auto queue = cl::CommandQueue(context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);
    cl::Buffer buf_a(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR, x.bsize(), x.inner());
    uint64_t m = 1;
    uint64_t s = stride == 0 ? SOKUDO_OPENCL_BLAS_SCAMAX_STRIDE : stride;
    auto local_size = wgs == 0 ? SOKUDO_OPENCL_BLAS_SCAMAX_WGS : wgs;
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
    auto task = new CLTask(queue);
    detail(task, "BLAS_SCAMAX", x.name(), incx.name(), res.name(), local_size, s);
    return task;
}

sokudo::CLTask *
sokudo::opencl::kernels::blas::cl_dcamax(const sokudo::Buffer<double2> &x, const sokudo::Value<uint64_t> &incx,
                                         const sokudo::Value<double> &res, uint64_t wgs, uint64_t stride) {
    // register kernel
    register_amax();

    auto kernel = KernelProvider::get(sokudo::KERNEL_BLAS_AMAX, "dcamax");
    auto context = kernel.getInfo<CL_KERNEL_CONTEXT>();
    auto device = context.getInfo<CL_CONTEXT_DEVICES>().front();

    uint64_t n = x.size();

    auto queue = cl::CommandQueue(context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);
    cl::Buffer buf_a(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR, x.bsize(), x.inner());
    uint64_t m = 1;
    uint64_t s = stride == 0 ? SOKUDO_OPENCL_BLAS_DCAMAX_STRIDE : stride;
    auto local_size = wgs == 0 ? SOKUDO_OPENCL_BLAS_DCAMAX_WGS : wgs;
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

    auto task = new CLTask(queue);
    detail(task, "BLAS_DCAMAX", x.name(), incx.name(), res.name(), local_size, s);
    return task;
}
