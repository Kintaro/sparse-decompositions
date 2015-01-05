#ifndef EXPR_TRANSPOSE_EXPR_H_
#define EXPR_TRANSPOSE_EXPR_H_

#include "conjugate_transpose_expr.h"

namespace expr {

template <typename ExprType, typename ValueType>
using TransposeExpr = ConjugateTransposeExpr<ExprType, ValueType, internal::Conjugator<false, ValueType>>;

} // namespace expr

#endif  // EXPR_TRANSPOSE_EXPR_H_
