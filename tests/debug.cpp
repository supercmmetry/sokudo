#include <gtest/gtest.h>
#include <sokudo_cuda/cuda_test.h>
#include <sokudo.h>

class DebugTest : public testing::Test {};

TEST(DebugTest, CudaSample) {
    int *a = new int[10];
    int *b = new int[10];

    for (int i = 0; i < 10; i++) {
        a[i] = b[i] = i;
    }

    auto buf_a = sokudo::DataBuffer(a, 10);
    auto buf_b = sokudo::DataBuffer(b, 10);

    auto task = cu_add_test(buf_a.inner(), buf_b.inner(), 10);
    task.sync();

    for (int i = 0; i < 10; i++) {
        ASSERT_EQ(buf_b[i], i * 2);
    }
}