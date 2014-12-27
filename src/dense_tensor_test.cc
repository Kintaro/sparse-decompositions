#include <gtest/gtest.h>
#include <limits>

#include "dense_tensor.h"

namespace {

// A large integer used to produce overflows.
constexpr auto kLargeInteger = std::numeric_limits<std::size_t>::max();

// Wrapper function to create a tensor of the given dimensions.
template <std::size_t P>
void AllocateTensor(const typename DenseTensor<float, P>::Index& size) {
  DenseTensor<float, P> x(size);
}

}  // namespace

TEST(DenseTensor, OverflowSafeMultiplicationSanityCheck) {
  // Don't die if either operand is zero.
  EXPECT_EQ(0, internal::MultiplyOrDie<std::size_t>(0, kLargeInteger));
  EXPECT_EQ(0, internal::MultiplyOrDie<std::size_t>(kLargeInteger, 0));

  // Should die on overflow.
  ASSERT_DEATH(internal::MultiplyOrDie<std::size_t>(kLargeInteger, kLargeInteger), ".*");
}

TEST(DenseTensor, IndexHelperSanityCheck) {
  // The linearized index of (0, ..., 0) must be 0 for any number of modes.
  std::size_t index = internal::IndexHelper<1, 0>::Run({{0}}, {{1}});
  EXPECT_EQ(0, index);
  index = internal::IndexHelper<2, 1>::Run({{0, 0}}, {{1, 2}});
  EXPECT_EQ(0, index);
  index = internal::IndexHelper<3, 2>::Run({{0, 0, 0}}, {{1, 2, 6}});
  EXPECT_EQ(0, index);
  index = internal::IndexHelper<4, 3>::Run({{0, 0, 0, 0}}, {{1, 2, 6, 4}});
  EXPECT_EQ(0, index);

  // Linearize indices for a matrix should be row-major.
  index = internal::IndexHelper<2, 1>::Run({{0, 0}}, {{2, 2}});
  EXPECT_EQ(0, index);
  index = internal::IndexHelper<2, 1>::Run({{1, 0}}, {{2, 2}});
  EXPECT_EQ(1, index);
  index = internal::IndexHelper<2, 1>::Run({{0, 1}}, {{2, 2}});
  EXPECT_EQ(2, index);
  index = internal::IndexHelper<2, 1>::Run({{1, 1}}, {{2, 2}});
  EXPECT_EQ(3, index);
}

TEST(DenseTensor, ConstructorShouldDieOnEmptyModes) {
  ASSERT_DEATH(AllocateTensor<2>({{0, 0}}), ".*");
}

TEST(DenseTensor, ConstructorShouldDieOnOverflow) {
  ASSERT_DEATH(AllocateTensor<2>({{kLargeInteger, kLargeInteger}}), ".*");
}

TEST(DenseTensor, GetShouldDieOnIndexOutOfBounds) {
  DenseTensor<float, 2> tensor({{2, 2}});
  EXPECT_DEATH(tensor.Get({{2, 1}}), ".*");
  EXPECT_DEATH(tensor.Get({{1, 2}}), ".*");
}

TEST(DenseTensor, SetShouldDieOnIndexOutOfBounds) {
  DenseTensor<float, 2> tensor({{2, 2}});
  EXPECT_DEATH(tensor.Set({{2, 1}}, 0.0f), ".*");
  EXPECT_DEATH(tensor.Set({{1, 2}}, 0.0f), ".*");
}

TEST(DenseTensor, GetSetSanityCheck) {
  DenseTensor<float, 2> tensor({{8, 8}});

  EXPECT_EQ(&tensor, &tensor.Set({{0, 0}}, 1.0f));
  EXPECT_EQ(1.0f, tensor.Get({{0, 0}}));

  tensor.Set({{3, 7}}, -14.5f);
  EXPECT_EQ(-14.5f, tensor.Get({{3, 7}}));
}

TEST(DenseTensor, SubscriptOperatorShouldDieOnIndexOutOfBounds) {
  const DenseTensor<float, 2> tensor({{2, 2}});
  DenseTensor<float, 2> writable_tensor({{2, 2}});

  EXPECT_DEATH(tensor.operator[]({{2, 1}}), ".*");
  EXPECT_DEATH(tensor.operator[]({{1, 2}}), ".*");
  EXPECT_DEATH(writable_tensor.operator[]({{2, 1}}), ".*");
  EXPECT_DEATH(writable_tensor.operator[]({{1, 2}}), ".*");
}

TEST(DenseTensor, SubscriptOperatorSanityCheck) {
  DenseTensor<float, 2> tensor({{8, 8}});

  tensor[{{0, 0}}] = 1.0f;
  EXPECT_EQ(1.0f, (tensor[{{0, 0}}]));

  tensor[{{6, 4}}] = -6.3f;
  EXPECT_EQ(-6.3f, (tensor[{{6, 4}}]));
}

TEST(DenseTensor, ComputeNormSanityCheck) {
  // The zero tensor should have norm zero.
  DenseTensor<float, 2> tensor({{2, 2}});
  EXPECT_EQ(0, tensor.ComputeNorm());

  tensor[{{0, 0}}] = 1.0f;
  tensor[{{0, 1}}] = 2.0f;
  tensor[{{1, 0}}] = 3.0f;
  tensor[{{1, 1}}] = 4.0f;
  EXPECT_FLOAT_EQ(5.477225575, tensor.ComputeNorm());

  tensor[{{0, 1}}] *= -1.0f;
  EXPECT_FLOAT_EQ(5.477225575, tensor.ComputeNorm());
}

TEST(DenseTensor, DimensionsSanityCheck) {
  DenseTensor<float, 6> tensor({{1, 2, 3, 4, 5, 6}});
  EXPECT_EQ(1, tensor.dimensions()[0]);
  EXPECT_EQ(2, tensor.dimensions()[1]);
  EXPECT_EQ(3, tensor.dimensions()[2]);
  EXPECT_EQ(4, tensor.dimensions()[3]);
  EXPECT_EQ(5, tensor.dimensions()[4]);
  EXPECT_EQ(6, tensor.dimensions()[5]);
}

TEST(DenseTensor, SizeSanityCheck) {
  DenseTensor<float, 4> small_tensor({{1, 1, 1, 1}});
  EXPECT_EQ(1, small_tensor.size());

  DenseTensor<float, 8> tensor({{2, 2, 2, 2, 2, 2, 2, 2}});
  EXPECT_EQ(256, tensor.size());
}
