#include <gtest/gtest.h>

#include "expr/transpose_expr.h"
#include "tensor/storage_mode.h"
#include "tensor/tensor.h"

using expr::TransposeExpr;
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

TEST(TransposeExprTest, SanityCheck) {
  // Create a tensor
  Matrix<float> tensor = CreateMatrix();
  auto transposed = !tensor;
  // Transposing keeps the diagonal intact
  EXPECT_EQ(tensor.Get({{0, 0}}), transposed.Get({{0, 0}}));
  // 
  EXPECT_EQ(tensor.Get({{0, 1}}), transposed.Get({{1, 0}}));
  // Transposing two times results in an identity operation
  EXPECT_EQ(tensor.Get({{0, 1}}), (!!tensor).Get({{0, 1}}));
}

}  // namespace test
