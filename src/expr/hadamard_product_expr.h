#ifndef EXPR_HADAMARD_PRODUCT_EXPR_H_
#define EXPR_HADAMARD_PRODUCT_EXPR_H_

#include "expr/expr_util.h"
#include "expr/tensor_expr.h"

namespace expr {

template <typename FirstExprType, typename SecondExprType, typename ValueType, std::size_t Order>
class HadamardProductExpr
    : public TensorExpr<HadamardProductExpr<FirstExprType, SecondExprType, ValueType, Order>,
                        ValueType, Order> {
 public:
  using type = HadamardProductExpr<FirstExprType, SecondExprType, ValueType, Order>;
  using base_type = TensorExpr<type, ValueType, Order>;
  using size_type = typename base_type::size_type;
  using pos_type = typename base_type::pos_type;
  using value_type = typename base_type::value_type;

  // Creates a Hadamard product expression representing the coefficient-wise product of two tensors
  // of identical dimensions.
  HadamardProductExpr(const FirstExprType& first_expr, const SecondExprType& second_expr)
      : first_expr_(first_expr), second_expr_(second_expr) {
    CHECK(util::container_equals(first_expr_.dimensions(), second_expr_.dimensions()));
  }

  pos_type dimensions() const override {
    return first_expr_.dimensions();
  }

  size_type size() const override {
    return first_expr_.size();
  }

  const value_type Get(const pos_type& index) const override {
    return first_expr_.Get(index) * second_expr_.Get(index);
  }

  // TODO(mnett,kintaro): Implement this once we have the capability to materialize expressions as
  // dense or sparse tensors.
  value_type abs() const override {
    CHECK(false) << "Not implemented yet!";
    return value_type{0};    
  }

 private:
  const FirstExprType& first_expr_;
  const SecondExprType& second_expr_;
};

}  // namespace expr

template <typename FirstExprType, typename SecondExprType, typename ValueType, std::size_t Order>
const expr::HadamardProductExpr<FirstExprType, SecondExprType, ValueType, Order> operator*(
    const expr::TensorExpr<FirstExprType, ValueType, Order>& first_expr,
    const expr::TensorExpr<SecondExprType, ValueType, Order>& second_expr) {
  return expr::HadamardProductExpr<FirstExprType, SecondExprType, ValueType, Order>(first_expr,
                                                                                    second_expr);
}

#endif  // EXPR_HADAMARD_PRODUCT_EXPR_H_