#include <gtest/gtest.h>

class DebugTest : public testing::Test {};

TEST(DebugTest, Test1) {
    std::cout << "hello" << std::endl;
}