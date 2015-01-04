#ifndef EXPR_SCALAR_PRODUCT_EXPR_H_
#define EXPR_SCALAR_PRODUCT_EXPR_H_

#include "expr/tensor_expr.h"
#include "tensor/storage_mode.h"

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

  pos_type dimensions() const override {
    return expr_.dimensions();
  }

  size_type size() const override {
    return expr_.size();
  }

  const value_type Get(const pos_type& index) const override {
    return scalar_ * expr_.Get(index);
  }

  value_type abs() const override {
    CHECK(false) << "Not implemented yet!";
    return value_type{0};
  }

  tensor::StorageMode GetPreferredStorageMode() const override {
    return expr_.GetPreferredStorageMode();
  }

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

#endif  // EXPR_SCALAR_PRODUCT_EXPR_H_
