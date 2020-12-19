#include "cl_test.h"
#include <CL/cl2.hpp>
#include <vector>
#include <iostream>

void cl_platform_test() {
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    std::cout << "OpenCL platforms found: " << platforms.size() << std::endl;

    for (const auto& platform : platforms) {
        std::vector<cl::Device> devices;
        platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
        std::cout << "Devices found: " << devices.size() << std::endl;

        for (const auto& device : devices) {
            std::cout << "\tDevice details:" << std::endl;
            std::cout << "\t\tVendor: " << device.getInfo<CL_DEVICE_VENDOR>() << std::endl;
            std::cout << "\t\tVersion: " << device.getInfo<CL_DEVICE_VERSION>() << std::endl;
        }
    }
}

void cl_add_test(int *a, int *b, int n) {

}

