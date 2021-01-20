#include <gtest/gtest.h>
#include <sokudo.h>
#include <types.h>

#ifdef SOKUDO_CUDA
#include <sokudo_cuda/blas/level1/axpy.h>
#endif

#ifdef SOKUDO_OPENCL
#include <sokudo_opencl/blas/level1/axpy.h>
#endif

class DebugTest : public testing::Test {};

TEST(DebugTest, Debug) {
    auto a = new sokudo::float2[1048576];
    for (int i = 0; i < 1048576; i++) {
        a[i].x = (float)(1);
        a[i].y = (float)(-1);
    }

    sokudo::float2 alpha = {1.0, 1.0};

    auto x = sokudo::Buffer(a, 1000);
    auto y = sokudo::Buffer(a, 1000);
    sokudo::opencl::DeviceProvider::load_devices();
    auto task = cu_scaxpy(&alpha, a, 1, a, 1, 1000, 1000, 1000);
    task.sync();
    task.destroy();

    std::cout << a[0].x << ", " << a[0].y << std::endl;
}