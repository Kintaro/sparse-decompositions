namespace expr {

template <typename ExprType, typename ValueType, std::size_t Order>
typename TensorExpr<ExprType, ValueType, Order>::pos_type
TensorExpr<ExprType, ValueType, Order>::dimensions() const {
  return static_cast<const ExprType&>(*this).dimensions();
}

template <typename ExprType, typename ValueType, std::size_t Order>
typename TensorExpr<ExprType, ValueType, Order>::size_type
TensorExpr<ExprType, ValueType, Order>::size() const {
  return static_cast<const ExprType&>(*this).size();
}

template <typename ExprType, typename ValueType, std::size_t Order>
const typename TensorExpr<ExprType, ValueType, Order>::value_type&
TensorExpr<ExprType, ValueType, Order>::
operator[](const pos_type& i) const {
  return static_cast<const ExprType&>(*this)[i];
}

}  // namespace expr
