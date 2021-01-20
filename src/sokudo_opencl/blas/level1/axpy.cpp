#include "axpy.h"

#ifndef SOKUDO_OPENCL_BLAS_SAXPY_WGS
#define SOKUDO_OPENCL_BLAS_SAXPY_WGS kernel.getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(device)
#endif

#ifndef SOKUDO_OPENCL_BLAS_SAXPY_STRIDE
#define SOKUDO_OPENCL_BLAS_SAXPY_STRIDE 0x10
#endif

#ifndef SOKUDO_OPENCL_BLAS_DAXPY_WGS
#define SOKUDO_OPENCL_BLAS_DAXPY_WGS kernel.getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(device)
#endif

#ifndef SOKUDO_OPENCL_BLAS_DAXPY_STRIDE
#define SOKUDO_OPENCL_BLAS_DAXPY_STRIDE 0x10
#endif

#ifndef SOKUDO_OPENCL_BLAS_SCAXPY_WGS
#define SOKUDO_OPENCL_BLAS_SCAXPY_WGS kernel.getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(device)
#endif

#ifndef SOKUDO_OPENCL_BLAS_SCAXPY_STRIDE
#define SOKUDO_OPENCL_BLAS_SCAXPY_STRIDE 0x10
#endif

#ifndef SOKUDO_OPENCL_BLAS_DCAXPY_WGS
#define SOKUDO_OPENCL_BLAS_DCAXPY_WGS kernel.getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(device)
#endif

#ifndef SOKUDO_OPENCL_BLAS_DCAXPY_STRIDE
#define SOKUDO_OPENCL_BLAS_DCAXPY_STRIDE 0x10
#endif

void sokudo::opencl::kernels::blas::register_axpy() {
    sokudo::opencl::ProgramProvider::register_kernel(sokudo::KERNEL_BLAS_AXPY,

#include "axpy.cl"

    );
}

void detail_axpy_cl(
        sokudo::CLTask *task,
        const std::string &name,
        const std::string &n,
        const std::string &alpha,
        const std::string &x,
        const std::string &incx,
        const std::string &y,
        const std::string &incy,
        uint64_t wgs,
        uint64_t stride
) {
    task->set(sokudo::KERNEL_BLAS_AXPY, name);
    task->params()["n"] = n;
    task->params()["alpha"] = alpha;
    task->params()["x"] = x;
    task->params()["incx"] = incx;
    task->params()["y"] = y;
    task->params()["incy"] = incy;
    task->params()["wgs"] = std::to_string(wgs);
    task->params()["stride"] = std::to_string(stride);
}

sokudo::CLTask *
sokudo::opencl::kernels::blas::cl_saxpy(
        const sokudo::Value<uint64_t> &_n,
        const sokudo::Value<float> &alpha,
        const sokudo::Buffer<float> &x,
        const sokudo::Value<uint64_t> &incx,
        const sokudo::Buffer<float> &y,
        const sokudo::Value<uint64_t> &incy,
        uint64_t wgs,
        uint64_t stride
) {
    register_axpy();

    auto kernel = KernelProvider::get(sokudo::KERNEL_BLAS_AXPY, "saxpy");
    auto context = kernel.getInfo<CL_KERNEL_CONTEXT>();
    auto device = context.getInfo<CL_CONTEXT_DEVICES>().front();

    uint64_t n = _n.value();

    auto queue = cl::CommandQueue(context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);

    cl::Buffer buf_a(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, alpha.bsize(), alpha.inner());
    cl::Buffer buf_x(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, x.bsize(), x.inner());
    cl::Buffer buf_y(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR, y.bsize(), y.inner());
    uint64_t s = stride == 0 ? SOKUDO_OPENCL_BLAS_SAXPY_STRIDE : stride;
    auto local_size = wgs == 0 ? SOKUDO_OPENCL_BLAS_SAXPY_WGS : wgs;
    auto global_size = (n / s + (n % s != 0));
    global_size = (global_size / local_size + (global_size % local_size != 0)) * local_size;

    kernel.setArg(0, buf_a);
    kernel.setArg(1, buf_x);
    kernel.setArg(2, incx.value());
    kernel.setArg(3, buf_y);
    kernel.setArg(4, incy.value());
    kernel.setArg(5, n);
    kernel.setArg(6, s);

    queue.enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(global_size), cl::NDRange(local_size));
    queue.enqueueBarrierWithWaitList();
    queue.enqueueReadBuffer(buf_y, CL_FALSE, 0, y.bsize(), y.inner());

    auto task = new CLTask(queue);
    detail_axpy_cl(task, "BLAS_SAXPY", _n.name(), alpha.name(), x.name(), incx.name(), y.name(), incy.name(), local_size, s);
    return task;
}

sokudo::CLTask *
sokudo::opencl::kernels::blas::cl_daxpy(
        const sokudo::Value<uint64_t> &_n,
        const sokudo::Value<double> &alpha,
        const sokudo::Buffer<double> &x,
        const sokudo::Value<uint64_t> &incx,
        const sokudo::Buffer<double> &y,
        const sokudo::Value<uint64_t> &incy,
        uint64_t wgs,
        uint64_t stride
) {
    register_axpy();

    auto kernel = KernelProvider::get(sokudo::KERNEL_BLAS_AXPY, "daxpy");
    auto context = kernel.getInfo<CL_KERNEL_CONTEXT>();
    auto device = context.getInfo<CL_CONTEXT_DEVICES>().front();

    uint64_t n = _n.value();

    auto queue = cl::CommandQueue(context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);

    cl::Buffer buf_a(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, alpha.bsize(), alpha.inner());
    cl::Buffer buf_x(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, x.bsize(), x.inner());
    cl::Buffer buf_y(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR, y.bsize(), y.inner());
    uint64_t s = stride == 0 ? SOKUDO_OPENCL_BLAS_SAXPY_STRIDE : stride;
    auto local_size = wgs == 0 ? SOKUDO_OPENCL_BLAS_SAXPY_WGS : wgs;
    auto global_size = (n / s + (n % s != 0));
    global_size = (global_size / local_size + (global_size % local_size != 0)) * local_size;

    kernel.setArg(0, buf_a);
    kernel.setArg(1, buf_x);
    kernel.setArg(2, incx.value());
    kernel.setArg(3, buf_y);
    kernel.setArg(4, incy.value());
    kernel.setArg(5, n);
    kernel.setArg(6, s);

    queue.enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(global_size), cl::NDRange(local_size));
    queue.enqueueBarrierWithWaitList();
    queue.enqueueReadBuffer(buf_y, CL_FALSE, 0, y.bsize(), y.inner());

    auto task = new CLTask(queue);
    detail_axpy_cl(task, "BLAS_DAXPY", _n.name(), alpha.name(), x.name(), incx.name(), y.name(), incy.name(), local_size, s);
    return task;
}

sokudo::CLTask *
sokudo::opencl::kernels::blas::cl_scaxpy(
        const sokudo::Value<uint64_t> &_n,
        const sokudo::Value<float2> &alpha,
        const sokudo::Buffer<float2> &x,
        const sokudo::Value<uint64_t> &incx,
        const sokudo::Buffer<float2> &y,
        const sokudo::Value<uint64_t> &incy,
        uint64_t wgs,
        uint64_t stride
) {
    register_axpy();

    auto kernel = KernelProvider::get(sokudo::KERNEL_BLAS_AXPY, "scaxpy");
    auto context = kernel.getInfo<CL_KERNEL_CONTEXT>();
    auto device = context.getInfo<CL_CONTEXT_DEVICES>().front();

    uint64_t n = _n.value();

    auto queue = cl::CommandQueue(context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);

    cl::Buffer buf_a(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, alpha.bsize(), alpha.inner());
    cl::Buffer buf_x(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, x.bsize(), x.inner());
    cl::Buffer buf_y(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR, y.bsize(), y.inner());
    uint64_t s = stride == 0 ? SOKUDO_OPENCL_BLAS_SAXPY_STRIDE : stride;
    auto local_size = wgs == 0 ? SOKUDO_OPENCL_BLAS_SAXPY_WGS : wgs;
    auto global_size = (n / s + (n % s != 0));
    global_size = (global_size / local_size + (global_size % local_size != 0)) * local_size;

    kernel.setArg(0, buf_a);
    kernel.setArg(1, buf_x);
    kernel.setArg(2, incx.value());
    kernel.setArg(3, buf_y);
    kernel.setArg(4, incy.value());
    kernel.setArg(5, n);
    kernel.setArg(6, s);

    queue.enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(global_size), cl::NDRange(local_size));
    queue.enqueueBarrierWithWaitList();
    queue.enqueueReadBuffer(buf_y, CL_FALSE, 0, y.bsize(), y.inner());

    auto task = new CLTask(queue);
    detail_axpy_cl(task, "BLAS_SCAXPY", _n.name(), alpha.name(), x.name(), incx.name(), y.name(), incy.name(), local_size, s);
    return task;
}

sokudo::CLTask *
sokudo::opencl::kernels::blas::cl_dcaxpy(
        const sokudo::Value<uint64_t> &_n,
        const sokudo::Value<double2> &alpha,
        const sokudo::Buffer<double2> &x,
        const sokudo::Value<uint64_t> &incx,
        const sokudo::Buffer<double2> &y,
        const sokudo::Value<uint64_t> &incy,
        uint64_t wgs,
        uint64_t stride
) {
    register_axpy();

    auto kernel = KernelProvider::get(sokudo::KERNEL_BLAS_AXPY, "dcaxpy");
    auto context = kernel.getInfo<CL_KERNEL_CONTEXT>();
    auto device = context.getInfo<CL_CONTEXT_DEVICES>().front();

    uint64_t n = _n.value();

    auto queue = cl::CommandQueue(context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);

    cl::Buffer buf_a(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, alpha.bsize(), alpha.inner());
    cl::Buffer buf_x(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, x.bsize(), x.inner());
    cl::Buffer buf_y(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR, y.bsize(), y.inner());
    uint64_t s = stride == 0 ? SOKUDO_OPENCL_BLAS_SAXPY_STRIDE : stride;
    auto local_size = wgs == 0 ? SOKUDO_OPENCL_BLAS_SAXPY_WGS : wgs;
    auto global_size = (n / s + (n % s != 0));
    global_size = (global_size / local_size + (global_size % local_size != 0)) * local_size;

    kernel.setArg(0, buf_a);
    kernel.setArg(1, buf_x);
    kernel.setArg(2, incx.value());
    kernel.setArg(3, buf_y);
    kernel.setArg(4, incy.value());
    kernel.setArg(5, n);
    kernel.setArg(6, s);

    queue.enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(global_size), cl::NDRange(local_size));
    queue.enqueueBarrierWithWaitList();
    queue.enqueueReadBuffer(buf_y, CL_FALSE, 0, y.bsize(), y.inner());

    auto task = new CLTask(queue);
    detail_axpy_cl(task, "BLAS_DCAXPY", _n.name(), alpha.name(), x.name(), incx.name(), y.name(), incy.name(), local_size, s);
    return task;
}
