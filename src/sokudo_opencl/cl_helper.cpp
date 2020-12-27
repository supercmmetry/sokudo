#include "cl_helper.h"
#include <errors.h>

#ifndef SOKUDO_OPENCL_BUILD_OPTIONS
#define SOKUDO_OPENCL_BUILD_OPTIONS "-cl-std=CL2.0"
#endif

using namespace sokudo::opencl;

std::unordered_map<sokudo::Kernel, cl::Program> ProgramProvider::_program_map;
std::mutex ProgramProvider::_mutex;

std::vector<cl::Device> DeviceProvider::_devices;
std::mutex DeviceProvider::_mutex;
uint64_t DeviceProvider::_device_index = 0;


cl::Program ProgramProvider::get(sokudo::Kernel kernel) {
    _mutex.lock();
    if (!_program_map.contains(kernel)) {
        _mutex.unlock();
        throw sokudo::errors::ResolutionException("Failed to load un-registered OpenCL kernel");
    }
    _mutex.unlock();
    return _program_map[kernel];
}

void ProgramProvider::clear() {
    _mutex.lock();
    _program_map.clear();
    _mutex.unlock();
}

void ProgramProvider::register_kernel(sokudo::Kernel kernel, const std::string &src) {
    _mutex.lock();
    if (!_program_map.contains(kernel)) {
        cl::Context context(DeviceProvider::get());
        auto program = cl::Program(context, src);
        program.build(SOKUDO_OPENCL_BUILD_OPTIONS);
        _program_map[kernel] = program;
    }
    _mutex.unlock();
}

cl::Kernel KernelProvider::get(sokudo::Kernel kernel) {
    cl::Program program = ProgramProvider::get(kernel);
    return cl::Kernel(program, "run");
}
