#include <gtest/gtest.h>
#include <sokudo.h>

class TaskGroupTest : public testing::Test {
};

TEST(TaskGroupTest, TaskGroup1) {
#ifdef SOKUDO_OPENCL
    sokudo::opencl::DeviceProvider::load_devices();
#endif

    auto a = new float[1048576];
    for (int i = 0; i < 1048576; i++) {
        a[i] = 1.0;
    }

    auto buf_a = sokudo::Buffer(a, 1048576);
    auto res = sokudo::Value<float>(0);
    auto incx = sokudo::Value<uint64_t>(1);
    sokudo::TaskGroup()
            .add(sokudo::kernels::blas::Asum<sokudo::CUDA>()(buf_a, incx, res))
            .add(sokudo::kernels::blas::Asum<sokudo::CUDA>()(buf_a, incx, res))
            .add(sokudo::kernels::blas::Asum<sokudo::CUDA>()(buf_a, incx, res))
            .add(sokudo::kernels::blas::Asum<sokudo::OPENCL>()(buf_a, incx, res))
            .add(sokudo::kernels::blas::Asum<sokudo::CUDA>()(buf_a, incx, res))
            .add(sokudo::kernels::blas::Asum<sokudo::CUDA>()(buf_a, incx, res))
            .add(sokudo::kernels::blas::Asum<sokudo::CUDA>()(buf_a, incx, res))
            .then([&]() -> sokudo::TaskGroup {
                      return sokudo::TaskGroup()
                              .add(sokudo::kernels::blas::Asum<sokudo::OPENCL>()(buf_a, incx, res))
                              .add(sokudo::kernels::blas::Asum<sokudo::CUDA>()(buf_a, incx, res))
                              .add(sokudo::kernels::blas::Asum<sokudo::OPENCL>()(buf_a, incx, res))
                              .then([&]() -> sokudo::TaskGroup {
                                  return sokudo::TaskGroup()
                                          .add(sokudo::kernels::blas::Asum<sokudo::OPENCL>()(buf_a, incx, res))
                                          .add(sokudo::kernels::blas::Asum<sokudo::CUDA>()(buf_a, incx, res))
                                          .add(sokudo::kernels::blas::Asum<sokudo::CUDA>()(buf_a, incx, res));
                              });
                  }
            )
            .sync();


    ASSERT_TRUE(res == (float) 1048576.0);

    delete[] a;
}
