#include "cl_test.h"
#include <CL/cl2.hpp>
#include <vector>
#include <iostream>

using namespace sokudo::opencl;

void cl_platform_test() {
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    std::cout << "OpenCL platforms found: " << platforms.size() << std::endl;

    for (const auto &platform : platforms) {
        std::vector<cl::Device> devices;
        platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
        std::cout << "Devices found: " << devices.size() << std::endl;

        for (const auto &device : devices) {
            std::cout << "\tDevice details:" << std::endl;
            std::cout << "\t\tVendor: " << device.getInfo<CL_DEVICE_VENDOR>() << std::endl;
            std::cout << "\t\tVersion: " << device.getInfo<CL_DEVICE_VERSION>() << std::endl;
        }
    }
}

sokudo::CLTask* kernels::cl_add_test(const sokudo::DataBuffer<int> &a, const sokudo::DataBuffer<int> &b) {
    auto kernel = KernelProvider::get(sokudo::TEST);
    auto context = kernel.getInfo<CL_KERNEL_CONTEXT>();
    auto device = context.getInfo<CL_CONTEXT_DEVICES>().front();


    auto queue = cl::CommandQueue(context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);


    cl::Buffer buf_a(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, a.bsize(), a.inner());
    cl::Buffer buf_b(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR, b.bsize(), b.inner());

    kernel.setArg(0, buf_a);
    kernel.setArg(1, buf_b);
    queue.enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(a.size()));
    queue.enqueueReadBuffer(buf_b, CL_FALSE, 0, b.bsize(), b.inner());

    return new CLTask(queue);
}


