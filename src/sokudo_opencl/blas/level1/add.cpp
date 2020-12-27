#include <kernel_enums.h>
#include <sokudo_opencl/cl_helper.h>
#include "add.h"

sokudo::CLTask *sokudo::opencl::kernels::blas::cl_math_add_real(
        const sokudo::DataBuffer<int64_t> &a,
        const sokudo::DataBuffer<int64_t> &b,
        const sokudo::DataBuffer<int64_t> &c
) {
    // register kernel
    sokudo::opencl::ProgramProvider::register_kernel(sokudo::KERNEL_MATH_ADD_I64,
#include "add_i64.cl"
    );

    auto kernel = KernelProvider::get(sokudo::KERNEL_MATH_ADD_I64);
    auto context = kernel.getInfo<CL_KERNEL_CONTEXT>();
    auto device = context.getInfo<CL_CONTEXT_DEVICES>().front();

    uint64_t n = a.size();
    auto local_size = kernel.getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(device);
    auto global_size = n / local_size + (n % local_size != 0);

    auto queue = cl::CommandQueue(context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);

    cl::Buffer buf_a(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, a.bsize(), a.inner());
    cl::Buffer buf_b(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, b.bsize(), b.inner());
    cl::Buffer buf_c(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY, c.bsize());

    kernel.setArg(0, buf_a);
    kernel.setArg(1, buf_b);
    kernel.setArg(2, buf_c);
    kernel.setArg(3, n);

    queue.enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(global_size), cl::NDRange(local_size));
    queue.enqueueReadBuffer(buf_c, CL_FALSE, 0, c.bsize(), c.inner());
    return new CLTask(queue);
}
