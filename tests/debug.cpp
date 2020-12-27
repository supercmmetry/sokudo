#include <gtest/gtest.h>
#include <sokudo.h>

class DebugTest : public testing::Test {};

#ifdef SOKUDO_CUDA
TEST(DebugTest, CudaSample) {
    int *a = new int[10];
    int *b = new int[10];

    for (int i = 0; i < 10; i++) {
        a[i] = b[i] = i;
    }

    auto buf_a = sokudo::DataBuffer(a, 10);
    auto buf_b = sokudo::DataBuffer(b, 10);

    auto task = sokudo::kernels::AddTest<sokudo::CUDA>()(buf_a, buf_b);
    task->sync();

    for (int i = 0; i < 10; i++) {
        ASSERT_EQ(buf_b[i], i * 2);
    }
}
#endif

#ifdef SOKUDO_OPENCL
TEST(DebugTest, OpenCLSample) {
    int *a = new int[10];
    for (int i = 0; i < 10; i++) {
        a[i] = i;
    }

    sokudo::opencl::DeviceProvider::load_devices();

    auto buf_a = sokudo::DataBuffer(a, 10);
    auto buf_b = sokudo::DataBuffer(a, 10);

    auto task = sokudo::kernels::AddTest<sokudo::OPENCL>()(buf_a, buf_b);
    task->sync();

    for (int i = 0; i < 10; i++) {
        ASSERT_EQ(buf_b[i], i * 2);
    }
}
#endif