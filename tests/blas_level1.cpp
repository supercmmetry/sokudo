#include <gtest/gtest.h>
#include <sokudo.h>

class BlasLevel1Test : public testing::Test {};

#ifdef SOKUDO_CUDA

#endif

#ifdef SOKUDO_OPENCL
TEST(BlasLevel1Test, OpenCLSasum) {
    auto a = new float[1048576];
    for (int i = 0; i < 1048576; i++) {
        a[i] = 1.0;
    }

    sokudo::opencl::DeviceProvider::load_devices();

    auto buf_a = sokudo::DataBuffer(a, 1048576);
    auto res = sokudo::DataValue((float)0);

    auto task = sokudo::kernels::blas::Sasum<sokudo::OPENCL>()(buf_a, res);
    task->sync();
    ASSERT_TRUE(res == (float) 1048576.0);
}
#endif