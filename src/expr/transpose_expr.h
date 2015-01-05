#ifndef EXPR_TRANSPOSE_EXPR_H_
#define EXPR_TRANSPOSE_EXPR_H_

#include "conjugate_transpose_expr.h"

namespace expr {

template <typename ExprType, typename ValueType>
using TransposeExpr = ConjugateTransposeExpr<ExprType, ValueType, internal::Conjugator<false, ValueType>>;

} // namespace expr

template <typename ExprType, typename ValueType>
const expr::TransposeExpr<ExprType, ValueType> operator!(
    const expr::MatrixExpr<ExprType, ValueType>& expr) {
  return expr::TransposeExpr<ExprType, ValueType>(expr);
}

#endif  // EXPR_TRANSPOSE_EXPR_H_
