#include <gtest/gtest.h>
#include <sokudo.h>

class BlasLevel1Test : public testing::Test {};

#ifdef SOKUDO_CUDA
TEST(BlasLevel1Test, CudaAddInt) {

}
#endif

#ifdef SOKUDO_OPENCL
TEST(BlasLevel1Test, OpenCLAddInt) {
    auto a = new int64_t[1048576];
    auto b = new int64_t[1048576];
    auto c = new int64_t[1048576];
    for (int i = 0; i < 1048576; i++) {
        a[i] = b[i] = i;
    }

    sokudo::opencl::DeviceProvider::load_devices();

    auto buf_a = sokudo::DataBuffer(a, 1048576);
    auto buf_b = sokudo::DataBuffer(b, 1048576);
    auto buf_c = sokudo::DataBuffer(c, 1048576);

    auto task = sokudo::kernels::blas::AddInt<sokudo::OPENCL>()(buf_a, buf_b, buf_c);
    task->sync();

    for (int i = 0; i < 1048576; i++) {
        ASSERT_EQ(buf_c[i], i * 2);
    }

    delete[] a;
    delete[] b;
    delete[] c;
}
#endif