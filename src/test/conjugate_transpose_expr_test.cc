#include <gtest/gtest.h>

#include "expr/conjugate_transpose_expr.h"
#include "tensor/storage_mode.h"
#include "tensor/tensor.h"

using expr::ConjugateTransposeExpr;
using tensor::StorageMode;
using tensor::Matrix;

namespace test {
namespace {

Matrix<float> CreateMatrix() {
  Matrix<float> tensor({{2, 2}}, StorageMode::DENSE);
  tensor.Set({{0, 0}}, 1.0f);
  tensor.Set({{0, 1}}, 2.0f);
  tensor.Set({{1, 0}}, 3.0f);
  tensor.Set({{1, 1}}, 4.0f);
  return tensor;
}

}  // namespace

TEST(ConjugateTransposeExprTest, SanityCheck) {
  // Create a tensor and verify its norm.
  Matrix<float> tensor = CreateMatrix();
  const auto conj = ~tensor;
  // EXPECT_FLOAT_EQ(5.477225575, tensor.abs());

  // // Scaling a tensor by 1 or -1 must preserve its norm.
  // {
  //   auto pos_scaled = 1.0f * tensor;
  //   EXPECT_FLOAT_EQ(5.477225575, pos_scaled.abs());
  //   auto neg_scaled = -1.0f * tensor;
  //   EXPECT_FLOAT_EQ(5.477225575, neg_scaled.abs());
  // }

  // // Tensor scaling should be positively homogenous.
  // {
  //   auto pos_scaled = 2.0f * tensor;
  //   EXPECT_FLOAT_EQ(2.0f * 5.477225575f, pos_scaled.abs());
  //   auto neg_scaled = -2.0f * tensor;
  //   EXPECT_FLOAT_EQ(2.0f * 5.477225575f, neg_scaled.abs());
  // }
}

}  // namespace test
