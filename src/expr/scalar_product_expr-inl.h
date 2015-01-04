#include <cmath>

namespace expr {

template <typename ExprType, typename ValueType, std::size_t Order>
ScalarProductExpr<ExprType, ValueType, Order>::ScalarProductExpr(const ValueType& scalar,
                                                                 const ExprType& expr)
    : scalar_(scalar), expr_(expr) {}

template <typename ExprType, typename ValueType, std::size_t Order>
typename ScalarProductExpr<ExprType, ValueType, Order>::pos_type
ScalarProductExpr<ExprType, ValueType, Order>::dimensions() const {
  return expr_.dimensions();
}

template <typename ExprType, typename ValueType, std::size_t Order>
typename ScalarProductExpr<ExprType, ValueType, Order>::size_type
ScalarProductExpr<ExprType, ValueType, Order>::size() const {
  return expr_.size();
}

template <typename ExprType, typename ValueType, std::size_t Order>
const typename ScalarProductExpr<ExprType, ValueType, Order>::value_type
ScalarProductExpr<ExprType, ValueType, Order>::
operator[](const pos_type& i) const {
  return scalar_ * expr_[i];
}

template <typename ExprType, typename ValueType, std::size_t Order>
typename ScalarProductExpr<ExprType, ValueType, Order>::value_type
ScalarProductExpr<ExprType, ValueType, Order>::abs() const {
  return std::abs(scalar_) * expr_.abs();
}

}  // namespace expr
