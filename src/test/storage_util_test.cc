#include <gtest/gtest.h>

#include "tensor/storage_util.h"

namespace test {
namespace {

// A large integer used to produce overflows.
constexpr auto kLargeInteger = std::numeric_limits<std::size_t>::max();

}  // namespace

TEST(StorageUtilTest, OverflowSafeMultiplicationSanityCheck) {
  // Don't die if either operand is zero.
  EXPECT_EQ(0, tensor::MultiplyOrDie<std::size_t>(0, kLargeInteger));
  EXPECT_EQ(0, tensor::MultiplyOrDie<std::size_t>(kLargeInteger, 0));

  // Should die on overflow.
  EXPECT_DEATH(tensor::MultiplyOrDie<std::size_t>(kLargeInteger, kLargeInteger),
               ".*");
}

TEST(StorageUtilTest, IndexHelperSanityCheck) {
  // The linearized index of (0, ..., 0) must be 0 for any number of modes.
  std::size_t index = tensor::IndexHelper<1, 0>::Run({{0}}, {{1}});
  EXPECT_EQ(0, index);
  index = tensor::IndexHelper<2, 1>::Run({{0, 0}}, {{1, 2}});
  EXPECT_EQ(0, index);
  index = tensor::IndexHelper<3, 2>::Run({{0, 0, 0}}, {{1, 2, 6}});
  EXPECT_EQ(0, index);
  index = tensor::IndexHelper<4, 3>::Run({{0, 0, 0, 0}}, {{1, 2, 6, 4}});
  EXPECT_EQ(0, index);

  // Linearize indices for a matrix should be row-major.
  index = tensor::IndexHelper<2, 1>::Run({{0, 0}}, {{2, 2}});
  EXPECT_EQ(0, index);
  index = tensor::IndexHelper<2, 1>::Run({{1, 0}}, {{2, 2}});
  EXPECT_EQ(1, index);
  index = tensor::IndexHelper<2, 1>::Run({{0, 1}}, {{2, 2}});
  EXPECT_EQ(2, index);
  index = tensor::IndexHelper<2, 1>::Run({{1, 1}}, {{2, 2}});
  EXPECT_EQ(3, index);
}

TEST(StorageUtilTest, LinearIndexShouldBeIdentityOnFirstOrderTensors) {
  std::size_t index = tensor::IndexHelper<1, 0>::Run({{0}}, {{10}});
  EXPECT_EQ(0, index);
  index = tensor::IndexHelper<1, 0>::Run({{7}}, {{10}});
  EXPECT_EQ(7, index);
  index = tensor::IndexHelper<1, 0>::Run({{9}}, {{10}});
  EXPECT_EQ(9, index);
}

TEST(StorageUtilTest, ReverseIndexHelperSanityCheck) {
  auto index = tensor::ReverseIndexHelper<2, 1>::Run(2, {{2, 2}});
  auto expected = std::array<std::uint64_t, 2>({{0, 1}});
  EXPECT_EQ(expected, index);
  index = tensor::ReverseIndexHelper<2, 1>::Run(1, {{2, 2}});
  expected = std::array<std::uint64_t, 2>({{1, 0}});
  EXPECT_EQ(expected, index);
}

TEST(StorageUtilTest, ReverseLinearIndexShouldBeIdentityOnFirstOrderTensors) {
  auto index = tensor::ReverseIndexHelper<1, 0>::Run(0, {{10}});
  auto expected = std::array<std::uint64_t, 1>({{0}});
  EXPECT_EQ(expected, index);
  expected = std::array<std::uint64_t, 1>({{7}});
  index = tensor::ReverseIndexHelper<1, 0>::Run(7, {{10}});
  EXPECT_EQ(expected, index);
  expected = std::array<std::uint64_t, 1>({{9}});
  index = tensor::ReverseIndexHelper<1, 0>::Run(9, {{10}});
  EXPECT_EQ(expected, index);
}

}  // namespace test
