#ifndef EXPR_MATRIX_EXPR_H_
#define EXPR_MATRIX_EXPR_H_

#include <array>
#include <cstddef>
#include <glog/logging.h>

#include "tensor_expr.h"

namespace expr {
  template <typename ExprType, typename ValueType>
  using MatrixExpr = TensorExpr<ExprType, ValueType, 2>;
}  // namespace expr

#endif  // EXPR_TENSOR_EXPR_H_
