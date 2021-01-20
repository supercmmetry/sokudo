#include <gtest/gtest.h>
#include <sokudo.h>
#include <sokudo_cuda/blas/level1/axpy.h>

class DebugTest : public testing::Test {};

TEST(DebugTest, CudaSample) {
    auto a = new float[1048576];
    for (int i = 0; i < 1048576; i++) {
        a[i] = (float)(i % 289);
    }

    float alpha = 1.0;

    auto task = cu_saxpy(&alpha, a, 1, a, 1, 1000);
    task.sync();
    task.destroy();

    for (int i = 0; i < 1000; i++) {
        ASSERT_EQ(a[i], 2 * (float)(i % 289));
    }

}