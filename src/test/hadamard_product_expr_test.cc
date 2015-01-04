#include <gtest/gtest.h>

#include "expr/hadamard_product_expr.h"
#include "tensor/storage_mode.h"
#include "tensor/tensor.h"

using expr::HadamardProductExpr;
using tensor::StorageMode;
using tensor::Tensor;

namespace test {
namespace {

Tensor<float, 2> CreateTensor() {
  Tensor<float, 2> tensor({{2, 2}}, StorageMode::DENSE);
  tensor.Set({{0, 0}}, 1.0f);
  tensor.Set({{0, 1}}, 2.0f);
  tensor.Set({{1, 0}}, 3.0f);
  tensor.Set({{1, 1}}, 4.0f);
  return tensor;
}

}  // namespace

// TODO(mnett): Revive once abs() is implemented.
TEST(HadamardProductExprTest, SanityCheck) {
  // Create a 2x2-tensor filled with ones.
  Tensor<float, 2> ones({{2, 2}}, StorageMode::DENSE);
  ones.Set({{0, 0}}, 1.0f).Set({{0, 1}}, 1.0f).Set({{1, 0}}, 1.0f).Set({{1, 1}}, 1.0f);

  // Create two 2x2-tensors filled with mixed values.
  Tensor<float, 2> mixed({{2, 2}}, StorageMode::DENSE);
  mixed.Set({{0, 0}}, 0.0f).Set({{0, 1}}, 1.0f).Set({{1, 0}}, -1.0f).Set({{1, 1}}, 2.0f);
  Tensor<float, 2> more_mixed({{2, 2}}, StorageMode::DENSE);
  more_mixed.Set({{0, 0}}, 3.0f).Set({{0, 1}}, -2.0f).Set({{1, 0}}, -3.0f).Set({{1, 1}}, 4.0f);

  // Create a 2x3-tensor filled with zeros.
  Tensor<float, 2> zeros({{2, 3}}, StorageMode::DENSE);

  // Verify that the Hadamard expression dies on tensors of different sizes.
  EXPECT_DEATH((ones * zeros), ".*");
  EXPECT_DEATH((mixed * zeros), ".*");

  // Verify that the Hadamard product implements coefficient-wise multiplication.
  const auto expr = mixed * more_mixed;
  EXPECT_EQ(0.0f, expr.Get({{0, 0}}));
  EXPECT_EQ(-2.0f, expr.Get({{0, 1}}));
  EXPECT_EQ(3.0f, expr.Get({{1, 0}}));
  EXPECT_EQ(8.0f, expr.Get({{1, 1}}));
}

}  // namespace test