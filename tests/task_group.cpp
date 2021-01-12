#include <gtest/gtest.h>
#include <sokudo.h>

class TaskGroupTest : public testing::Test {
};

TEST(TaskGroupTest, TaskGroup1) {
    auto a = new float[1048576];
    for (int i = 0; i < 1048576; i++) {
        a[i] = 1.0;
    }

    auto buf_a = sokudo::DataBuffer(a, 1048576);
    auto res = sokudo::DataValue<float>(0);
    auto incx = sokudo::DataValue<uint64_t>(1);
    sokudo::TaskGroup() <<
                        sokudo::kernels::blas::Sasum<sokudo::CUDA>()(buf_a, incx, res) <<
                        sokudo::kernels::blas::Sasum<sokudo::CUDA>()(buf_a, incx, res) <<
                        sokudo::kernels::blas::Sasum<sokudo::CUDA>()(buf_a, incx, res) >>
                        sokudo::TaskGroup() <<
                        sokudo::kernels::blas::Sasum<sokudo::OPENCL>()(buf_a, incx, res) <<
                        sokudo::kernels::blas::Sasum<sokudo::CUDA>()(buf_a, incx, res) <<
                        sokudo::kernels::blas::Sasum<sokudo::OPENCL>()(buf_a, incx, res) <<
                        sokudo::TaskGroup::SYNC;

    sokudo::TaskGroup()
            .add(sokudo::kernels::blas::Sasum<sokudo::CUDA>()(buf_a, incx, res))
            .add(sokudo::kernels::blas::Sasum<sokudo::CUDA>()(buf_a, incx, res))
            .add(sokudo::kernels::blas::Sasum<sokudo::CUDA>()(buf_a, incx, res))
            .add(sokudo::kernels::blas::Sasum<sokudo::CUDA>()(buf_a, incx, res))
            .then(
                    sokudo::TaskGroup()
                            .add(sokudo::kernels::blas::Sasum<sokudo::OPENCL>()(buf_a, incx, res))
                            .add(sokudo::kernels::blas::Sasum<sokudo::CUDA>()(buf_a, incx, res))
                            .add(sokudo::kernels::blas::Sasum<sokudo::OPENCL>()(buf_a, incx, res))
                            .then([&]() {
                                return sokudo::TaskGroup()
                                        .add(sokudo::kernels::blas::Sasum<sokudo::OPENCL>()(buf_a, incx, res))
                                        .add(sokudo::kernels::blas::Sasum<sokudo::CUDA>()(buf_a, incx, res))
                                        .add(sokudo::kernels::blas::Sasum<sokudo::CUDA>()(buf_a, incx, res));
                            }())
            )
            .sync();


    ASSERT_TRUE(res == (float) 1048576.0);

    delete[] a;
}
