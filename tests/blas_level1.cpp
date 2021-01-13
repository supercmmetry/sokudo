#include <gtest/gtest.h>
#include <sokudo.h>

using namespace sokudo;

class BlasLevel1Test : public testing::Test {};

#ifdef SOKUDO_CUDA
TEST(BlasLevel1Test, CUDASasum1) {
    auto a = new float[1048576];
    for (int i = 0; i < 1048576; i++) {
        a[i] = 1.0;
    }

    auto buf_a = sokudo::Buffer(a, 1048576);
    auto res = sokudo::Value<float>(0);
    auto incx = sokudo::Value<uint64_t>(1);

    auto task = sokudo::kernels::blas::Asum<sokudo::CUDA>()(buf_a, incx, res);
    task->sync();
    ASSERT_TRUE(res == (float) 1048576.0);

    delete[] a;
}

TEST(BlasLevel1Test, CUDASasum2) {
    auto a = new float[1048576];
    for (int i = 0; i < 1048576; i++) {
        a[i] = 1.0;
    }

    auto buf_a = sokudo::Buffer(a, 1048576);
    auto res = sokudo::Value<float>(0);
    auto incx = sokudo::Value<uint64_t>(2);

    auto task = sokudo::kernels::blas::Asum<sokudo::CUDA>()(buf_a, incx, res);
    task->sync();
    ASSERT_TRUE(res == (float) 524288);

    delete[] a;
}

TEST(BlasLevel1Test, CUDADasum1) {
    auto a = new double[1048576];
    for (int i = 0; i < 1048576; i++) {
        a[i] = 1.0;
    }

    auto buf_a = sokudo::Buffer(a, 1048576);
    auto res = sokudo::Value<double>(0);
    auto incx = sokudo::Value<uint64_t>(1);

    auto task = sokudo::kernels::blas::Asum<sokudo::CUDA>()(buf_a, incx, res);
    task->sync();

    ASSERT_TRUE(res == 1048576.0);
    delete[] a;
}

TEST(BlasLevel1Test, CUDADasum2) {
    auto a = new double[1048576];
    for (int i = 0; i < 1048576; i++) {
        a[i] = 1.0;
    }

    auto buf_a = sokudo::Buffer(a, 1048576);
    auto res = sokudo::Value<double>(0);
    auto incx = sokudo::Value<uint64_t>(2);

    auto task = sokudo::kernels::blas::Asum<sokudo::CUDA>()(buf_a, incx, res);
    task->sync();

    ASSERT_TRUE(res == 524288.0);
    delete[] a;
}

TEST(BlasLevel1Test, CUDAScasum) {
    auto a = new float2[1048576];
    for (int i = 0; i < 1048576; i++) {
        a[i] = float2{ .x=1.0, .y=-1.0 };
    }

    auto buf_a = sokudo::Buffer(a, 1048576);
    auto res = sokudo::Value<float>(0);
    auto incx = sokudo::Value<uint64_t>(2);

    auto task = sokudo::kernels::blas::Asum<sokudo::CUDA>()(buf_a, incx, res);
    task->sync();

    ASSERT_TRUE(res == 1048576.0);
    delete[] a;
}

TEST(BlasLevel1Test, CUDADcasum) {
    auto a = new double2[1048576];
    for (int i = 0; i < 1048576; i++) {
        a[i] = { 1.0, -1.0 };
    }

    auto buf_a = sokudo::Buffer(a, 1048576);
    auto res = sokudo::Value<double>(0);
    auto incx = sokudo::Value<uint64_t>(2);

    auto task = sokudo::kernels::blas::Asum<sokudo::CUDA>()(buf_a, incx, res);
    task->sync();

    ASSERT_TRUE(res == 1048576.0);
    delete[] a;
}
#endif

#ifdef SOKUDO_OPENCL
TEST(BlasLevel1Test, OpenCLSasum1) {
    auto a = new float[1048576];
    for (int i = 0; i < 1048576; i++) {
        a[i] = 1.0;
    }

    sokudo::opencl::DeviceProvider::load_devices();

    auto buf_a = sokudo::Buffer(a, 1048576);
    auto incx = sokudo::Value<uint64_t>(1);
    auto res = sokudo::Value<float>(0);

    auto task = sokudo::kernels::blas::Asum<sokudo::OPENCL>()(buf_a, incx, res);
    task->sync();
    ASSERT_TRUE(res == (float) 1048576.0);

    delete[] a;
}

TEST(BlasLevel1Test, OpenCLSasum2) {
    auto a = new float[1048576];
    for (int i = 0; i < 1048576; i++) {
        a[i] = 1.0;
    }

    sokudo::opencl::DeviceProvider::load_devices();

    auto buf_a = sokudo::Buffer(a, 1048576);
    auto incx = sokudo::Value<uint64_t>(2);
    auto res = sokudo::Value<float>(0);

    auto task = sokudo::kernels::blas::Asum<sokudo::OPENCL>()(buf_a, incx, res);
    task->sync();
    ASSERT_TRUE(res == (float) 524288);

    delete[] a;
}

TEST(BlasLevel1Test, OpenCLDasum1) {
    auto a = new double[1048576];
    for (int i = 0; i < 1048576; i++) {
        a[i] = 1.0;
    }

    sokudo::opencl::DeviceProvider::load_devices();

    auto buf_a = sokudo::Buffer(a, 1048576);
    auto incx = sokudo::Value<uint64_t>(1);
    auto res = sokudo::Value<double>(0);

    auto task = sokudo::kernels::blas::Asum<sokudo::OPENCL>()(buf_a, incx, res);
    task->sync();
    ASSERT_TRUE(res == 1048576.0);

    delete[] a;
}

TEST(BlasLevel1Test, OpenCLDasum2) {
    sokudo::opencl::DeviceProvider::load_devices();

    auto a = new double[1048576];
    for (int i = 0; i < 1048576; i++) {
        a[i] = 1.0;
    }

    auto buf_a = sokudo::Buffer(a, 1048576);
    auto res = sokudo::Value<double>(0);
    auto incx = sokudo::Value<uint64_t>(2);

    auto task = sokudo::kernels::blas::Asum<sokudo::OPENCL>()(buf_a, incx, res);
    task->sync();

    ASSERT_TRUE(res == 524288.0);
    delete[] a;
}

TEST(BlasLevel1Test, OpenCLScasum) {
    sokudo::opencl::DeviceProvider::load_devices();

    auto a = new float2[1048576];
    for (int i = 0; i < 1048576; i++) {
        a[i] = float2{ .x=1.0, .y=-1.0 };
    }

    auto buf_a = sokudo::Buffer(a, 1048576);
    auto res = sokudo::Value<float>(0);
    auto incx = sokudo::Value<uint64_t>(2);

    auto task = sokudo::kernels::blas::Asum<sokudo::OPENCL>()(buf_a, incx, res);
    task->sync();

    ASSERT_TRUE(res == (float)1048576.0);
    delete[] a;
}

TEST(BlasLevel1Test, OpenCLDcasum) {
    sokudo::opencl::DeviceProvider::load_devices();

    auto a = new double2[1048576];
    for (int i = 0; i < 1048576; i++) {
        a[i] = { 1.0, -1.0 };
    }

    auto buf_a = sokudo::Buffer(a, 1048576).to_named("BUF_dcasum_1_input");
    auto res = sokudo::Value<double>(0).to_named("VAL_dcasum_1_output");
    auto incx = sokudo::Value<uint64_t>(2).to_named("VAL_dcasum_1_incx");

    auto task = sokudo::kernels::blas::Asum<sokudo::OPENCL>()(buf_a, incx, res);
    task->sync();

    ASSERT_TRUE(res == 1048576.0);
    delete[] a;
}

#endif