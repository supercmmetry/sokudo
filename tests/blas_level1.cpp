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

TEST(BlasLevel1Test, CUDAScasum1) {
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

TEST(BlasLevel1Test, CUDADcasum1) {
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

TEST(BlasLevel1Test, CUDASamax1) {
    auto a = new float[1048576];
    for (int i = 0; i < 1048576; i++) {
        a[i] = (float)(i % 289);
    }

    auto buf_a = sokudo::Buffer(a, 1048576);
    auto incx = sokudo::Value<uint64_t>(1);
    auto res = sokudo::Value<uint64_t>(0);

    auto task = sokudo::kernels::blas::Amax<sokudo::CUDA>()(buf_a, incx, res);
    task->sync();

    ASSERT_EQ(res.value(), 289);

    delete[] a;
}

TEST(BlasLevel1Test, CUDADamax1) {
    auto a = new double[1048576];
    for (int i = 0; i < 1048576; i++) {
        a[i] = (double)(i % 1);
    }

    auto buf_a = sokudo::Buffer(a, 1048576);
    auto incx = sokudo::Value<uint64_t>(1);
    auto res = sokudo::Value<uint64_t>(0);

    auto task = sokudo::kernels::blas::Amax<sokudo::CUDA>()(buf_a, incx, res);
    task->sync();

    ASSERT_EQ(res.value(), 1);

    delete[] a;
}

TEST(BlasLevel1Test, CUDAScamax1) {
    auto a = new float2[1048576];
    for (int i = 0; i < 1048576; i++) {
        a[i] = float2{ .x=(float)(i % 7), .y=-(float)(i % 9) };
    }

    auto buf_a = sokudo::Buffer(a, 1048576);
    auto res = sokudo::Value<uint64_t>(0);
    auto incx = sokudo::Value<uint64_t>(1);

    auto task = sokudo::kernels::blas::Amax<sokudo::CUDA>()(buf_a, incx, res);
    task->sync();

    ASSERT_EQ(res.value(), 63);
    delete[] a;
}

TEST(BlasLevel1Test, CUDADcamax1) {
    auto a = new double2[1048576];
    for (int i = 0; i < 1048576; i++) {
        a[i] = double2{ .x=(double)(i % 101), .y=-(double)(i % 103) };
    }

    auto buf_a = sokudo::Buffer(a, 1048576);
    auto res = sokudo::Value<uint64_t>(0);
    auto incx = sokudo::Value<uint64_t>(1);

    auto task = sokudo::kernels::blas::Amax<sokudo::CUDA>()(buf_a, incx, res);
    task->sync();

    ASSERT_EQ(res.value(), 10403);
    delete[] a;
}

TEST(BlasLevel1Test, CUDASamin1) {
    auto a = new float[1048576];
    for (int i = 0; i < 1048576; i++) {
        a[i] = (float)(289 - i % 289);
    }

    auto buf_a = sokudo::Buffer(a, 1048576);
    auto incx = sokudo::Value<uint64_t>(1);
    auto res = sokudo::Value<uint64_t>(0);

    auto task = sokudo::kernels::blas::Amin<sokudo::CUDA>()(buf_a, incx, res);
    task->sync();

    ASSERT_EQ(res.value(), 289);

    delete[] a;
}

TEST(BlasLevel1Test, CUDADamin1) {
    auto a = new double[1048576];
    for (int i = 0; i < 1048576; i++) {
        a[i] = (double)(1 - i % 1);
    }

    auto buf_a = sokudo::Buffer(a, 1048576);
    auto incx = sokudo::Value<uint64_t>(1);
    auto res = sokudo::Value<uint64_t>(0);

    auto task = sokudo::kernels::blas::Amin<sokudo::CUDA>()(buf_a, incx, res);
    task->sync();

    ASSERT_EQ(res.value(), 1);

    delete[] a;
}

TEST(BlasLevel1Test, CUDAScamin1) {
    auto a = new float2[1048576];
    for (int i = 0; i < 1048576; i++) {
        a[i] = float2{ .x=(float)(7 - i % 7), .y=-(float)(9 - i % 9) };
    }

    auto buf_a = sokudo::Buffer(a, 1048576);
    auto res = sokudo::Value<uint64_t>(0);
    auto incx = sokudo::Value<uint64_t>(1);

    auto task = sokudo::kernels::blas::Amin<sokudo::CUDA>()(buf_a, incx, res);
    task->sync();

    ASSERT_EQ(res.value(), 63);
    delete[] a;
}

TEST(BlasLevel1Test, CUDADcamin1) {
    auto a = new double2[1048576];
    for (int i = 0; i < 1048576; i++) {
        a[i] = double2{ .x=(double)(101 - i % 101), .y=-(double)(103 - i % 103) };
    }

    auto buf_a = sokudo::Buffer(a, 1048576);
    auto res = sokudo::Value<uint64_t>(0);
    auto incx = sokudo::Value<uint64_t>(1);

    auto task = sokudo::kernels::blas::Amin<sokudo::CUDA>()(buf_a, incx, res);
    task->sync();

    ASSERT_EQ(res.value(), 10403);
    delete[] a;
}

TEST(BlasLevel1Test, CUDASaxpy1) {

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

TEST(BlasLevel1Test, OpenCLScasum1) {
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

TEST(BlasLevel1Test, OpenCLDcasum1) {
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

TEST(BlasLevel1Test, OpenCLSamax1) {
    auto a = new float[2097152];
    for (int i = 0; i < 2097152; i++) {
        a[i] = (float)(i % 289);
    }

    sokudo::opencl::DeviceProvider::load_devices();

    auto buf_a = sokudo::Buffer(a, 2097152);
    auto incx = sokudo::Value<uint64_t>(1);
    auto res = sokudo::Value<uint64_t>(0);

    auto task = sokudo::kernels::blas::Amax<sokudo::OPENCL>()(buf_a, incx, res);
    task->sync();

    ASSERT_EQ(res.value(), 289);

    delete[] a;
}

TEST(BlasLevel1Test, OpenCLDamax1) {
    auto a = new double[1048576];
    for (int i = 0; i < 1048576; i++) {
        a[i] = (double)(i % 1);
    }

    sokudo::opencl::DeviceProvider::load_devices();

    auto buf_a = sokudo::Buffer(a, 1048576);
    auto incx = sokudo::Value<uint64_t>(1);
    auto res = sokudo::Value<uint64_t>(0);

    auto task = sokudo::kernels::blas::Amax<sokudo::OPENCL>()(buf_a, incx, res);
    task->sync();

    ASSERT_EQ(res.value(), 1);

    delete[] a;
}

TEST(BlasLevel1Test, OpenCLScamax1) {
    sokudo::opencl::DeviceProvider::load_devices();

    auto a = new float2[1048576];
    for (int i = 0; i < 1048576; i++) {
        a[i] = float2{ .x=(float)(i % 7), .y=-(float)(i % 9) };
    }

    auto buf_a = sokudo::Buffer(a, 1048576);
    auto res = sokudo::Value<uint64_t>(0);
    auto incx = sokudo::Value<uint64_t>(1);

    auto task = sokudo::kernels::blas::Amax<sokudo::OPENCL>()(buf_a, incx, res);
    task->sync();

    ASSERT_EQ(res.value(), 63);
    delete[] a;
}

TEST(BlasLevel1Test, OpenCLDcamax1) {
    sokudo::opencl::DeviceProvider::load_devices();

    auto a = new double2[1048576];
    for (int i = 0; i < 1048576; i++) {
        a[i] = double2{ .x=(double)(i % 101), .y=-(double)(i % 103) };
    }

    auto buf_a = sokudo::Buffer(a, 1048576);
    auto res = sokudo::Value<uint64_t>(0);
    auto incx = sokudo::Value<uint64_t>(1);

    auto task = sokudo::kernels::blas::Amax<sokudo::OPENCL>()(buf_a, incx, res);
    task->sync();

    ASSERT_EQ(res.value(), 10403);
    delete[] a;
}

TEST(BlasLevel1Test, OpenCLSamin1) {
    auto a = new float[1048576];
    for (int i = 0; i < 1048576; i++) {
        a[i] = (float)(289 - i % 289);
    }

    auto buf_a = sokudo::Buffer(a, 1048576);
    auto incx = sokudo::Value<uint64_t>(1);
    auto res = sokudo::Value<uint64_t>(0);

    auto task = sokudo::kernels::blas::Amin<sokudo::OPENCL>()(buf_a, incx, res);
    task->sync();

    ASSERT_EQ(res.value(), 289);

    delete[] a;
}

TEST(BlasLevel1Test, OpenCLDamin1) {
    auto a = new double[1048576];
    for (int i = 0; i < 1048576; i++) {
        a[i] = (double)(1 - i % 1);
    }

    sokudo::opencl::DeviceProvider::load_devices();

    auto buf_a = sokudo::Buffer(a, 1048576);
    auto incx = sokudo::Value<uint64_t>(1);
    auto res = sokudo::Value<uint64_t>(0);

    auto task = sokudo::kernels::blas::Amin<sokudo::OPENCL>()(buf_a, incx, res);
    task->sync();

    ASSERT_EQ(res.value(), 1);

    delete[] a;
}

TEST(BlasLevel1Test, OpenCLScamin1) {
    sokudo::opencl::DeviceProvider::load_devices();

    auto a = new float2[1048576];
    for (int i = 0; i < 1048576; i++) {
        a[i] = float2{ .x=(float)(7 - i % 7), .y=-(float)(9 - i % 9) };
    }

    auto buf_a = sokudo::Buffer(a, 1048576);
    auto res = sokudo::Value<uint64_t>(0);
    auto incx = sokudo::Value<uint64_t>(1);

    auto task = sokudo::kernels::blas::Amin<sokudo::OPENCL>()(buf_a, incx, res);
    task->sync();

    ASSERT_EQ(res.value(), 63);
    delete[] a;
}

TEST(BlasLevel1Test, OpenCLDcamin1) {
    sokudo::opencl::DeviceProvider::load_devices();

    auto a = new double2[1048576];
    for (int i = 0; i < 1048576; i++) {
        a[i] = double2{ .x=(double)(101 - i % 101), .y=-(double)(103 - i % 103) };
    }

    auto buf_a = sokudo::Buffer(a, 1048576);
    auto res = sokudo::Value<uint64_t>(0);
    auto incx = sokudo::Value<uint64_t>(1);

    auto task = sokudo::kernels::blas::Amin<sokudo::OPENCL>()(buf_a, incx, res);
    task->sync();

    ASSERT_EQ(res.value(), 10403);
    delete[] a;
}
#endif