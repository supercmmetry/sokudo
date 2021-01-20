#include <kernel_enums.h>
#include <sokudo_opencl/cl_helper.h>
#include <types.h>
#include "amin.h"

#ifndef SOKUDO_OPENCL_BLAS_SAMIN_WGS
#define SOKUDO_OPENCL_BLAS_SAMIN_WGS kernel.getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(device)
#endif

#ifndef SOKUDO_OPENCL_BLAS_SAMIN_STRIDE
#define SOKUDO_OPENCL_BLAS_SAMIN_STRIDE 0x10
#endif

#ifndef SOKUDO_OPENCL_BLAS_DAMIN_WGS
#define SOKUDO_OPENCL_BLAS_DAMIN_WGS kernel.getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(device)
#endif

#ifndef SOKUDO_OPENCL_BLAS_DAMIN_STRIDE
#define SOKUDO_OPENCL_BLAS_DAMIN_STRIDE 0x10
#endif

#ifndef SOKUDO_OPENCL_BLAS_SCAMIN_WGS
#define SOKUDO_OPENCL_BLAS_SCAMIN_WGS kernel.getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(device)
#endif

#ifndef SOKUDO_OPENCL_BLAS_SCAMIN_STRIDE
#define SOKUDO_OPENCL_BLAS_SCAMIN_STRIDE 0x10
#endif

#ifndef SOKUDO_OPENCL_BLAS_DCAMIN_WGS
#define SOKUDO_OPENCL_BLAS_DCAMIN_WGS kernel.getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(device)
#endif

#ifndef SOKUDO_OPENCL_BLAS_DCAMIN_STRIDE
#define SOKUDO_OPENCL_BLAS_DCAMIN_STRIDE 0x10
#endif

void sokudo::opencl::kernels::blas::register_amin() {
    sokudo::opencl::ProgramProvider::register_kernel(sokudo::KERNEL_BLAS_AMIN,

#include "amin.cl"

    );
}

void detail_amin_cl(
        sokudo::CLTask *task,
        const std::string &name,
        const std::string &n,
        const std::string &x,
        const std::string &incx,
        const std::string &res,
        uint64_t wgs,
        uint64_t stride
) {

    task->set(sokudo::KERNEL_BLAS_AMIN, name);
    task->params()["n"] = n;
    task->params()["x"] = x;
    task->params()["incx"] = incx;
    task->params()["res"] = res;
    task->params()["wgs"] = std::to_string(wgs);
    task->params()["stride"] = std::to_string(stride);
}

sokudo::CLTask *
sokudo::opencl::kernels::blas::cl_samin(const sokudo::Value<uint64_t> &_n, const sokudo::Buffer<float> &x,
                                        const sokudo::Value<uint64_t> &incx,
                                        const sokudo::Value<uint64_t> &res, uint64_t wgs, uint64_t stride) {
    // register kernel
    register_amin();

    auto kernel = KernelProvider::get(sokudo::KERNEL_BLAS_AMIN, "samin");
    auto context = kernel.getInfo<CL_KERNEL_CONTEXT>();
    auto device = context.getInfo<CL_CONTEXT_DEVICES>().front();

    uint64_t n = _n.value();

    auto queue = cl::CommandQueue(context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);

    cl::Buffer buf_a(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR, x.bsize(), x.inner());
    uint64_t m = 1;
    uint64_t s = stride == 0 ? SOKUDO_OPENCL_BLAS_SAMIN_STRIDE : stride;
    auto local_size = wgs == 0 ? SOKUDO_OPENCL_BLAS_SAMIN_WGS : wgs;
    auto global_size = (n / s + (n % s != 0));
    global_size = (global_size / local_size + (global_size % local_size != 0)) * local_size;

    cl::Buffer buf_b(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY, global_size * sizeof(uint64_t));

    while (m < n) {
        kernel.setArg(0, buf_a);
        kernel.setArg(1, s);
        kernel.setArg(2, n);
        kernel.setArg(3, m);
        kernel.setArg(4, incx.value());
        kernel.setArg(5, buf_b);

        m *= s;

        queue.enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(global_size), cl::NDRange(local_size));
        queue.enqueueBarrierWithWaitList();

        global_size = (global_size / s + (global_size % s != 0));
        global_size = (global_size / local_size + (global_size % local_size != 0)) * local_size;
    }

    queue.enqueueReadBuffer(buf_b, CL_FALSE, 0, res.bsize(), res.inner());

    auto task = new CLTask(queue);
    detail_amin_cl(task, "BLAS_SAMIN", _n.name(), x.name(), incx.name(), res.name(), local_size, s);
    return task;
}

sokudo::CLTask *
sokudo::opencl::kernels::blas::cl_damin(const sokudo::Value<uint64_t> &_n, const sokudo::Buffer<double> &x,
                                        const sokudo::Value<uint64_t> &incx,
                                        const sokudo::Value<uint64_t> &res, uint64_t wgs, uint64_t stride) {
    // register kernel
    register_amin();

    auto kernel = KernelProvider::get(sokudo::KERNEL_BLAS_AMIN, "damin");
    auto context = kernel.getInfo<CL_KERNEL_CONTEXT>();
    auto device = context.getInfo<CL_CONTEXT_DEVICES>().front();

    uint64_t n = _n.value();

    auto queue = cl::CommandQueue(context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);

    cl::Buffer buf_a(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR, x.bsize(), x.inner());
    uint64_t m = 1;
    uint64_t s = stride == 0 ? SOKUDO_OPENCL_BLAS_DAMIN_STRIDE : stride;
    auto local_size = wgs == 0 ? SOKUDO_OPENCL_BLAS_DAMIN_WGS : wgs;
    auto global_size = (n / s + (n % s != 0));
    global_size = (global_size / local_size + (global_size % local_size != 0)) * local_size;

    cl::Buffer buf_b(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY, global_size * sizeof(uint64_t));

    while (m < n) {
        kernel.setArg(0, buf_a);
        kernel.setArg(1, s);
        kernel.setArg(2, n);
        kernel.setArg(3, m);
        kernel.setArg(4, incx.value());
        kernel.setArg(5, buf_b);
        m *= s;

        queue.enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(global_size), cl::NDRange(local_size));
        queue.enqueueBarrierWithWaitList();

        global_size = (global_size / s + (global_size % s != 0));
        global_size = (global_size / local_size + (global_size % local_size != 0)) * local_size;
    }

    queue.enqueueReadBuffer(buf_b, CL_FALSE, 0, res.bsize(), res.inner());

    auto task = new CLTask(queue);
    detail_amin_cl(task, "BLAS_DAMIN", _n.name(), x.name(), incx.name(), res.name(), local_size, s);
    return task;
}

sokudo::CLTask *
sokudo::opencl::kernels::blas::cl_scamin(const sokudo::Value<uint64_t> &_n, const sokudo::Buffer<float2> &x,
                                         const sokudo::Value<uint64_t> &incx,
                                         const sokudo::Value<uint64_t> &res, uint64_t wgs, uint64_t stride) {
    // register kernel
    register_amin();

    auto kernel = KernelProvider::get(sokudo::KERNEL_BLAS_AMIN, "scamin");
    auto context = kernel.getInfo<CL_KERNEL_CONTEXT>();
    auto device = context.getInfo<CL_CONTEXT_DEVICES>().front();

    uint64_t n = _n.value();

    auto queue = cl::CommandQueue(context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);
    cl::Buffer buf_a(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR, x.bsize(), x.inner());
    uint64_t m = 1;
    uint64_t s = stride == 0 ? SOKUDO_OPENCL_BLAS_SCAMIN_STRIDE : stride;
    auto local_size = wgs == 0 ? SOKUDO_OPENCL_BLAS_SCAMIN_WGS : wgs;
    auto global_size = (n / s + (n % s != 0));
    global_size = (global_size / local_size + (global_size % local_size != 0)) * local_size;

    cl::Buffer buf_b(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY, global_size * sizeof(uint64_t));

    while (m < n) {
        kernel.setArg(0, buf_a);
        kernel.setArg(1, s);
        kernel.setArg(2, n);
        kernel.setArg(3, m);
        kernel.setArg(4, incx.value());
        kernel.setArg(5, buf_b);
        m *= s;

        queue.enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(global_size), cl::NDRange(local_size));
        queue.enqueueBarrierWithWaitList();

        global_size = (global_size / s + (global_size % s != 0));
        global_size = (global_size / local_size + (global_size % local_size != 0)) * local_size;
    }

    queue.enqueueReadBuffer(buf_b, CL_FALSE, 0, res.bsize(), res.inner());
    auto task = new CLTask(queue);
    detail_amin_cl(task, "BLAS_SCAMIN", _n.name(), x.name(), incx.name(), res.name(), local_size, s);
    return task;
}

sokudo::CLTask *
sokudo::opencl::kernels::blas::cl_dcamin(const sokudo::Value<uint64_t> &_n, const sokudo::Buffer<double2> &x,
                                         const sokudo::Value<uint64_t> &incx,
                                         const sokudo::Value<uint64_t> &res, uint64_t wgs, uint64_t stride) {
    // register kernel
    register_amin();

    auto kernel = KernelProvider::get(sokudo::KERNEL_BLAS_AMIN, "dcamin");
    auto context = kernel.getInfo<CL_KERNEL_CONTEXT>();
    auto device = context.getInfo<CL_CONTEXT_DEVICES>().front();

    uint64_t n = _n.value();

    auto queue = cl::CommandQueue(context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);
    cl::Buffer buf_a(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR, x.bsize(), x.inner());

    uint64_t m = 1;
    uint64_t s = stride == 0 ? SOKUDO_OPENCL_BLAS_DCAMIN_STRIDE : stride;
    auto local_size = wgs == 0 ? SOKUDO_OPENCL_BLAS_DCAMIN_WGS : wgs;
    auto global_size = (n / s + (n % s != 0));
    global_size = (global_size / local_size + (global_size % local_size != 0)) * local_size;

    cl::Buffer buf_b(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY, global_size * sizeof(uint64_t));
    while (m < n) {
        kernel.setArg(0, buf_a);
        kernel.setArg(1, s);
        kernel.setArg(2, n);
        kernel.setArg(3, m);
        kernel.setArg(4, incx.value());
        kernel.setArg(5, buf_b);
        m *= s;

        queue.enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(global_size), cl::NDRange(local_size));
        queue.enqueueBarrierWithWaitList();

        global_size = (global_size / s + (global_size % s != 0));
        global_size = (global_size / local_size + (global_size % local_size != 0)) * local_size;
    }

    queue.enqueueReadBuffer(buf_b, CL_FALSE, 0, res.bsize(), res.inner());

    auto task = new CLTask(queue);
    detail_amin_cl(task, "BLAS_DCAMIN", _n.name(), x.name(), incx.name(), res.name(), local_size, s);
    return task;
}
