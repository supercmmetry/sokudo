#include <gtest/gtest.h>
#include <sokudo.h>

class DebugTest : public testing::Test {};

TEST(DebugTest, CudaSample) {
    int *a = new int[10];
    int *b = new int[10];

    for (int i = 0; i < 10; i++) {
        a[i] = b[i] = i;
    }
    cu_add_test(a, b, 10);

    for (int i = 0; i < 10; i++) {
        ASSERT_EQ(b[i], i * 2);
    }
}

TEST(DebugTest, OpenCLSample) {
    cl_platform_test();
}