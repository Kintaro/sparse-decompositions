#include "expr/expr_util.h"

namespace expr {

template <typename FirstExprType, typename SecondExprType, typename ValueType, std::size_t Order>
HadamardProduct<FirstExprType, SecondExprType, ValueType, Order>::HadamardProduct(
    const FirstExprType& first_expr, const SecondExprType& second_expr)
    : first_expr_(first_expr), second_expr_(second_expr) {
  CHECK(util::container_equals(first_expr_.dimensions(), second_expr_.dimensions()));
}

template <typename FirstExprType, typename SecondExprType, typename ValueType, std::size_t Order>
typename HadamardProduct<FirstExprType, SecondExprType, ValueType, Order>::pos_type
HadamardProduct<FirstExprType, SecondExprType, ValueType, Order>::dimensions() const override {
  return first_expr_.dimensions();
}

template <typename FirstExprType, typename SecondExprType, typename ValueType, std::size_t Order>
typename HadamardProduct<FirstExprType, SecondExprType, ValueType, Order>::size_type
HadamardProduct<FirstExprType, SecondExprType, ValueType, Order>::size() const override {
  return first_expr_.size();
}

template <typename FirstExprType, typename SecondExprType, typename ValueType, std::size_t Order>
typename const HadamardProduct<FirstExprType, SecondExprType, ValueType, Order>::value_type
HadamardProduct<FirstExprType, SecondExprType, ValueType, Order>::
operator[](const pos_type& i) {
  return first_expr_[i] * second_expr_[i];
}

template <typename FirstExprType, typename SecondExprType, typename ValueType, std::size_t Order>
typename HadamardProduct<FirstExprType, SecondExprType, ValueType, Order>::value_type
HadamardProduct<FirstExprType, SecondExprType, ValueType, Order>::abs() const override {
  CHECK(false);  // Not implemented yet, let's worry about this when we have sparse tensors.
}

}  // namespace expr