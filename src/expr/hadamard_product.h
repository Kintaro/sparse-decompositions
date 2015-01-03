#ifndef EXPR_HADAMARD_PRODUCT_H_
#define EXPR_HADAMARD_PRODUCT_H_

#include "tensor_expr.h"

namespace expr {

template <typename FirstExprType, typename SecondExprType, typename ValueType, std::size_t Order>
class HadamardProduct
    : public TensorExpr<HadamardProduct<FirstExprType, SecondExprType, ValueType, Order>, ValueType,
                        Order> {
 public:
  using base_type = TensorExpr<HadamardProduct<FirstExprType, SecondExprType, ValueType, Order>,
                               ValueType, Order>;
  using size_type = typename base_type::size_type;
  using pos_type = typename base_type::pos_type;
  using value_type = typename base_type::value_type;

  // Creates a Hadamard product expression representing the coefficient-wise product of two tensors
  // of identical dimensions.
  HadamardProduct(const FirstExprType& first_expr, const SecondExprType& second_expr);

  pos_type dimensions() const override;
  size_type size() const override;
  const value_type& operator[](const pos_type& i) const override;

  // TODO(mnett,kintaro): Implement this once we have the capability to materialize expressions as
  // dense or sparse tensors.
  value_type abs() const override __attribute__((noreturn));

 private:
  const FirstExprType& first_expr_;
  const SecondExprType& second_expr_;
};

}  // namespace expr

#endif  // EXPR_HADAMARD_PRODUCT_H_