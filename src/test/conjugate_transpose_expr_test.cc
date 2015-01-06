#include <gtest/gtest.h>
#include <complex>

#include "expr/conjugate_transpose_expr.h"
#include "tensor/storage_mode.h"
#include "tensor/tensor.h"

using expr::ConjugateTransposeExpr;
using tensor::StorageMode;
using tensor::Matrix;

namespace test {
namespace {

Matrix<float> CreateRealMatrix() {
  Matrix<float> matrix({{3, 2}}, StorageMode::DENSE);
  matrix.Set({{0, 0}}, 1.0f);
  matrix.Set({{0, 1}}, 2.0f);
  matrix.Set({{1, 0}}, 3.0f);
  matrix.Set({{1, 1}}, 4.0f);
  matrix.Set({{2, 0}}, 7.0f);
  matrix.Set({{2, 1}}, 5.0f);
  return matrix;
}

Matrix<std::complex<float>> CreateComplexMatrix() {
  Matrix<std::complex<float>> matrix({{2, 3}}, StorageMode::DENSE);
  matrix.Set({{0, 0}}, std::complex<float>(1.0f, 1.0f));
  matrix.Set({{0, 1}}, std::complex<float>(5.0f, 2.0f));
  matrix.Set({{0, 2}}, std::complex<float>(3.0f, -1.0f));
  matrix.Set({{1, 0}}, std::complex<float>(1.0f, -3.0f));
  matrix.Set({{1, 1}}, std::complex<float>(-2.0f, 4.0f));
  matrix.Set({{1, 2}}, std::complex<float>(-1.0f, 2.0f));
  return matrix;
}

}  // namespace

TEST(ConjugateTransposeExprTest, ShouldConjugateTransposeComplexMatrix) {
  auto matrix = CreateComplexMatrix();
  auto transpose = ~matrix;

  ASSERT_EQ(2, transpose.dimensions().size());
  EXPECT_EQ(3, transpose.dimensions()[0]);
  EXPECT_EQ(2, transpose.dimensions()[1]);

  EXPECT_EQ(std::complex<float>(1.0f, -1.0f), transpose.Get({{0, 0}}));
  EXPECT_EQ(std::complex<float>(-1.0f, -2.0f), transpose.Get({{2, 1}}));
}

TEST(ConjugateTransposeExprTest, ConjugateTransposeShouldBeSelfInverse) {
  auto matrix = CreateComplexMatrix();
  auto same_matrix = ~~matrix;

  // TODO(mnett): Integrate google-mock to use container matchers.
  // TODO(mnett): Test against fixed values.
  ASSERT_EQ(2, matrix.dimensions().size());
  ASSERT_EQ(2, same_matrix.dimensions().size());
  EXPECT_EQ(matrix.dimensions()[0], same_matrix.dimensions()[0]);
  EXPECT_EQ(matrix.dimensions()[1], same_matrix.dimensions()[1]);

  EXPECT_EQ(matrix.Get({{0, 0}}), same_matrix.Get({{0, 0}}));
  EXPECT_EQ(matrix.Get({{0, 1}}), same_matrix.Get({{0, 1}}));
  EXPECT_EQ(matrix.Get({{0, 2}}), same_matrix.Get({{0, 2}}));
  EXPECT_EQ(matrix.Get({{1, 0}}), same_matrix.Get({{1, 0}}));
  EXPECT_EQ(matrix.Get({{1, 1}}), same_matrix.Get({{1, 1}}));
  EXPECT_EQ(matrix.Get({{1, 2}}), same_matrix.Get({{1, 2}}));
}

TEST(ConjugateTransposeExprTest, ShouldTransposeRealMatrix) {
  auto matrix = CreateRealMatrix();
  auto transpose = ~matrix;

  ASSERT_EQ(2, transpose.dimensions().size());
  EXPECT_EQ(2, transpose.dimensions()[0]);
  EXPECT_EQ(3, transpose.dimensions()[1]);

  EXPECT_EQ(1.0f, transpose.Get({{0, 0}}));
  EXPECT_EQ(5.0f, transpose.Get({{1, 2}}));
}

TEST(ConjugateTransposeExprTest, TransposeShouldBeSelfInverse) {
  auto matrix = CreateRealMatrix();
  auto same_matrix = ~~matrix;

  // TODO(mnett): Integrate google-mock to use container matchers.
  // TODO(mnett): Test against fixed values.
  ASSERT_EQ(2, matrix.dimensions().size());
  ASSERT_EQ(2, same_matrix.dimensions().size());
  EXPECT_EQ(matrix.dimensions()[0], same_matrix.dimensions()[0]);
  EXPECT_EQ(matrix.dimensions()[1], same_matrix.dimensions()[1]);

  EXPECT_EQ(matrix.Get({{0, 0}}), same_matrix.Get({{0, 0}}));
  EXPECT_EQ(matrix.Get({{0, 1}}), same_matrix.Get({{0, 1}}));
  EXPECT_EQ(matrix.Get({{1, 0}}), same_matrix.Get({{1, 0}}));
  EXPECT_EQ(matrix.Get({{1, 1}}), same_matrix.Get({{1, 1}}));
  EXPECT_EQ(matrix.Get({{2, 0}}), same_matrix.Get({{2, 0}}));
  EXPECT_EQ(matrix.Get({{2, 1}}), same_matrix.Get({{2, 1}}));
}

}  // namespace test
