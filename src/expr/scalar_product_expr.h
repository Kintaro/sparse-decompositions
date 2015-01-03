#ifndef EXPR_SCALAR_PRODUCT_EXPR_H_
#define EXPR_SCALAR_PRODUCT_EXPR_H_

#include "tensor_expr.h"

namespace expr {

template <typename ExprType, typename ValueType, std::size_t Order>
class ScalarProductExpr
    : public TensorExpr<ScalarProductExpr<ExprType, ValueType, Order>, ValueType, Order> {
 public:
  using base_type = TensorExpr<ScalarProductExpr<ExprType, ValueType, Order>, ValueType, Order>;
  using size_type = typename base_type::size_type;
  using pos_type = typename base_type::pos_type;
  using value_type = typename base_type::value_type;

  // Create a scalar product expression representing the product of a scalar and a tensor.
  ScalarProductExpr(const ValueType& scalar, const ExprType& expr);

  pos_type dimensions() const override;
  size_type size() const override;
  const value_type& operator[](const pos_type& i) const override;
  value_type abs() const override;

 private:
  const ValueType scalar_;
  const ExprType& expr_;
};

}  // namespace expr

template <typename ExprType, typename ValueType, std::size_t Order>
const expr::ScalarProductExpr<ExprType, ValueType, Order> operator*(
    const ValueType& scalar, const expr::TensorExpr<ExprType, ValueType, Order>& tensor) {
  return expr::ScalarProductExpr<ExprType, ValueType, Order>(scalar, tensor);
}

template <typename ExprType, typename ValueType, std::size_t Order>
const expr::ScalarProductExpr<ExprType, ValueType, Order> operator*(
    const expr::TensorExpr<ExprType, ValueType, Order>& tensor, const ValueType& scalar) {
  return expr::ScalarProductExpr<ExprType, ValueType, Order>(scalar, tensor);
}

#include "scalar_product_expr-inl.h"

#endif  // EXPR_SCALAR_PRODUCT_EXPR_H_
