#include <kernel_enums.h>
#include <sokudo_opencl/cl_helper.h>
#include "asum.h"

#ifndef SOKUDO_OPENCL_BLAS_SASUM_WGS
#define SOKUDO_OPENCL_BLAS_SASUM_WGS kernel.getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(device)
#endif

#ifndef SOKUDO_OPENCL_BLAS_SASUM_STRIDE
#define SOKUDO_OPENCL_BLAS_SASUM_STRIDE 0x10
#endif

sokudo::CLTask *sokudo::opencl::kernels::blas::cl_sasum(
        const sokudo::DataBuffer<float> &a,
        const sokudo::DataValue<float> &b
) {
    // register kernel
    sokudo::opencl::ProgramProvider::register_kernel(sokudo::KERNEL_BLAS_SASUM,
#include "sasum.cl"
    );

    auto kernel = KernelProvider::get(sokudo::KERNEL_BLAS_SASUM);
    auto context = kernel.getInfo<CL_KERNEL_CONTEXT>();
    auto device = context.getInfo<CL_CONTEXT_DEVICES>().front();

    uint64_t n = a.size();

    auto queue = cl::CommandQueue(context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);

    cl::Buffer buf_a(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR, a.bsize(), a.inner());
    uint64_t m = 1;
    uint64_t s = SOKUDO_OPENCL_BLAS_SASUM_STRIDE;
    auto local_size = SOKUDO_OPENCL_BLAS_SASUM_WGS;
    auto global_size = (n / s + (n % s != 0));
    global_size = (global_size / local_size + (global_size % local_size != 0)) * local_size;

    while (m < n) {
        kernel.setArg(0, buf_a);
        kernel.setArg(1, s);
        kernel.setArg(2, n);
        kernel.setArg(3, m);
        m *= s;

        queue.enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(global_size), cl::NDRange(local_size));
        queue.enqueueBarrierWithWaitList();

        global_size = (global_size / s + (global_size % s != 0));
        global_size = (global_size / local_size + (global_size % local_size != 0)) * local_size;
    }

    queue.enqueueReadBuffer(buf_a, CL_FALSE, 0, b.bsize(), b.inner());
    return new CLTask(queue);
}
