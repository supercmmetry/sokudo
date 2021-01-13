#include <gtest/gtest.h>
#include <sokudo.h>

class BlasLevel1Test : public testing::Test {};

#ifdef SOKUDO_CUDA
TEST(BlasLevel1Test, CUDASasum1) {
    auto a = new float[1048576];
    for (int i = 0; i < 1048576; i++) {
        a[i] = 1.0;
    }

    auto buf_a = sokudo::DataBuffer(a, 1048576);
    auto res = sokudo::DataValue<float>(0);
    auto incx = sokudo::DataValue<uint64_t>(1);

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

    auto buf_a = sokudo::DataBuffer(a, 1048576);
    auto res = sokudo::DataValue<float>(0);
    auto incx = sokudo::DataValue<uint64_t>(2);

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

    auto buf_a = sokudo::DataBuffer(a, 1048576);
    auto res = sokudo::DataValue<double>(0);
    auto incx = sokudo::DataValue<uint64_t>(1);

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

    auto buf_a = sokudo::DataBuffer(a, 1048576);
    auto res = sokudo::DataValue<double>(0);
    auto incx = sokudo::DataValue<uint64_t>(2);

    auto task = sokudo::kernels::blas::Asum<sokudo::CUDA>()(buf_a, incx, res);
    task->sync();

    ASSERT_TRUE(res == 524288.0);
    delete[] a;
}

TEST(BlasLevel1Test, CUDAScasum) {
    auto a = new float2[1048576];
    for (int i = 0; i < 1048576; i++) {
        a[i] = float2{ .x=1.0, .y=1.0 };
    }

    auto buf_a = sokudo::DataBuffer(a, 1048576);
    auto res = sokudo::DataValue<float>(0);
    auto incx = sokudo::DataValue<uint64_t>(2);

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

    auto buf_a = sokudo::DataBuffer(a, 1048576);
    auto incx = sokudo::DataValue<uint64_t>(1);
    auto res = sokudo::DataValue<float>(0);

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

    auto buf_a = sokudo::DataBuffer(a, 1048576);
    auto incx = sokudo::DataValue<uint64_t>(2);
    auto res = sokudo::DataValue<float>(0);

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

    auto buf_a = sokudo::DataBuffer(a, 1048576);
    auto incx = sokudo::DataValue<uint64_t>(1);
    auto res = sokudo::DataValue<double>(0);

    auto task = sokudo::kernels::blas::Asum<sokudo::OPENCL>()(buf_a, incx, res);
    task->sync();
    ASSERT_TRUE(res == 1048576.0);

    delete[] a;
}

TEST(BlasLevel1Test, OpenCLDasum2) {
    auto a = new double[1048576];
    for (int i = 0; i < 1048576; i++) {
        a[i] = 1.0;
    }

    auto buf_a = sokudo::DataBuffer(a, 1048576);
    auto res = sokudo::DataValue<double>(0);
    auto incx = sokudo::DataValue<uint64_t>(2);

    auto task = sokudo::kernels::blas::Asum<sokudo::OPENCL>()(buf_a, incx, res);
    task->sync();

    ASSERT_TRUE(res == 524288.0);
    delete[] a;
}

TEST(BlasLevel1Test, OpenCLScasum) {
    auto a = new float2[1048576];
    for (int i = 0; i < 1048576; i++) {
        a[i] = float2{ .x=1.0, .y=1.0 };
    }

    auto buf_a = sokudo::DataBuffer(a, 1048576);
    auto res = sokudo::DataValue<float>(0);
    auto incx = sokudo::DataValue<uint64_t>(2);

    auto task = sokudo::kernels::blas::Asum<sokudo::OPENCL>()(buf_a, incx, res);
    task->sync();

    ASSERT_TRUE(res == (float)1048576.0);
    delete[] a;
}
#endif