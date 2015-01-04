#include <gtest/gtest.h>

#include "tensor/dense_storage.h"

using tensor::DenseStorage;

namespace test {
namespace {

// A large integer used to produce overflows.
constexpr auto kLargeInteger = std::numeric_limits<std::size_t>::max();

}  // namespace

TEST(DenseStorageTest, OverflowSafeMultiplicationSanityCheck) {
  // Don't die if either operand is zero.
  EXPECT_EQ(0, tensor::internal::MultiplyOrDie<std::size_t>(0, kLargeInteger));
  EXPECT_EQ(0, tensor::internal::MultiplyOrDie<std::size_t>(kLargeInteger, 0));

  // // Should die on overflow.
  EXPECT_DEATH(tensor::internal::MultiplyOrDie<std::size_t>(kLargeInteger, kLargeInteger), ".*");
}

TEST(DenseStorageTest, IndexHelperSanityCheck) {
  // The linearized index of (0, ..., 0) must be 0 for any number of modes.
  std::size_t index = tensor::internal::IndexHelper<1, 0>::Run({{0}}, {{1}});
  EXPECT_EQ(0, index);
  index = tensor::internal::IndexHelper<2, 1>::Run({{0, 0}}, {{1, 2}});
  EXPECT_EQ(0, index);
  index = tensor::internal::IndexHelper<3, 2>::Run({{0, 0, 0}}, {{1, 2, 6}});
  EXPECT_EQ(0, index);
  index = tensor::internal::IndexHelper<4, 3>::Run({{0, 0, 0, 0}}, {{1, 2, 6, 4}});
  EXPECT_EQ(0, index);

  // Linearize indices for a matrix should be row-major.
  index = tensor::internal::IndexHelper<2, 1>::Run({{0, 0}}, {{2, 2}});
  EXPECT_EQ(0, index);
  index = tensor::internal::IndexHelper<2, 1>::Run({{1, 0}}, {{2, 2}});
  EXPECT_EQ(1, index);
  index = tensor::internal::IndexHelper<2, 1>::Run({{0, 1}}, {{2, 2}});
  EXPECT_EQ(2, index);
  index = tensor::internal::IndexHelper<2, 1>::Run({{1, 1}}, {{2, 2}});
  EXPECT_EQ(3, index);
}

TEST(DenseStorageTest, ConstructorShouldDieOnEmptyModes) {
  EXPECT_DEATH((DenseStorage<float, 2>({{0, 3}})), ".*");
  EXPECT_DEATH((DenseStorage<float, 2>({{2, 0}})), ".*");
  EXPECT_DEATH((DenseStorage<float, 1>({{0}})), ".*");
}

TEST(DenseStorageTest, ConstructorShouldDieOnOverflow) {
  EXPECT_DEATH((DenseStorage<float, 2>({{kLargeInteger, kLargeInteger}})), ".*");
}

TEST(DenseStorageTest, GetShouldDieOnIndexOutOfBounds) {
  DenseStorage<float, 2> storage({{2, 2}});
  EXPECT_DEATH((storage.Get({{2, 1}})), ".*");
  EXPECT_DEATH((storage.Get({{1, 2}})), ".*");
}

TEST(DenseStorageTest, SetShouldDieOnIndexOutOfBounds) {
  DenseStorage<float, 2> storage({{2, 2}});
  EXPECT_DEATH((storage.Set({{2, 1}}, 0.0f)), ".*");
  EXPECT_DEATH((storage.Set({{1, 2}}, 0.0f)), ".*");
}

TEST(DenseStorageTest, GetSetSanityCheck) {
  DenseStorage<float, 2> storage({{8, 8}});
  EXPECT_EQ(&storage, &storage.Set({{0, 0}}, -14.5f));
  EXPECT_EQ(-14.5f, storage.Get({{0, 0}}));
}

TEST(DenseStorageTest, DimensionsSanityCheck) {
  DenseStorage<float, 6> storage({{1, 3, 7, 41, 88, 4}});
  EXPECT_EQ(1, storage.dimensions()[0]);
  EXPECT_EQ(3, storage.dimensions()[1]);
  EXPECT_EQ(7, storage.dimensions()[2]);
  EXPECT_EQ(41, storage.dimensions()[3]);
  EXPECT_EQ(88, storage.dimensions()[4]);
  EXPECT_EQ(4, storage.dimensions()[5]);
}

TEST(DenseStorageTest, SizeSanityCheck) {
  DenseStorage<float, 4> singular_storage({{1,1,1,1}});
  EXPECT_EQ(1, singular_storage.size());

  DenseStorage<float, 8> storage({{2, 2, 2, 2, 2, 2, 2, 2}});
  EXPECT_EQ(256, storage.size());
}

}  // namespace test