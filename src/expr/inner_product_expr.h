#ifndef EXPR_INNER_PRODUCT_EXPR_H_
#define EXPR_INNER_PRODUCT_EXPR_H_

#include "tensor_expr.h"

namespace expr {

template <typename FirstExprType, typename SecondExprType, typename ValueType>
class InnerProductExpr
    : public TensorExpr<InnerProductExpr<FirstExprType, SecondExprType, ValueType>, ValueType, 1> {
 public:
  using base_type =
      TensorExpr<InnerProductExpr<FirstExprType, SecondExprType, ValueType>, ValueType, 1>;
  using size_type = typename base_type::size_type;
  using pos_type = typename base_type::pos_type;
  using value_type = typename base_type::value_type;

  // Create an inner product expression representing the product of a scalar and a tensor.
  InnerProductExpr(const FirstExprType& first_expr, const SecondExprType& second_expr);

 private:
  const FirstExprType& first_expr_;
  const SecondExprType& second_expr_;
};

}  // namespace expr

#endif  // EXPR_INNER_PRODUCT_EXPR_H_