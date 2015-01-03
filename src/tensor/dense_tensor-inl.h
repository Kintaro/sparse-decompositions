#ifndef DENSE_TENSOR_INL_H_
#define DENSE_TENSOR_INL_H_

namespace internal {

// Returns the product of two unsigned values. If the multiplication would create an overflow the
// function dies instead.
template <typename ValueType>
typename std::enable_if<std::is_arithmetic<ValueType>::value&& std::is_unsigned<ValueType>::value,
                        ValueType>::type
MultiplyOrDie(const ValueType& lhs, const ValueType& rhs) {
  if (lhs != ValueType(0)) {
    CHECK_GT(std::numeric_limits<ValueType>::max() / lhs, rhs);
  }
  return lhs * rhs;
}

// A variadic functor template to help with the linearization of a variable-length index into a
// tensor.
template <std::size_t Order, std::size_t N>
struct IndexHelper {
  constexpr static inline std::size_t Run(const std::array<std::uint64_t, Order>& index,
                                          const std::array<std::uint64_t, Order>& size) {
    return index[Order - N - 1] + size[Order - N - 1] * IndexHelper<Order, N - 1>::Run(index, size);
  }
};
template <std::size_t Order>
struct IndexHelper<Order, 0> {
  constexpr static inline std::size_t Run(const std::array<std::uint64_t, Order>& index,
                                          const std::array<std::uint64_t, Order>& size) {
    return index[Order - 1];
  }
};

}  // namespace internal

template <typename ValueType, std::size_t Order>
DenseTensor<ValueType, Order>::DenseTensor(const pos_type& dimensions)
    : dimensions_(dimensions) {
  std::for_each(dimensions_.cbegin(), dimensions_.cend(),
                [](const size_type& size) { CHECK_LT(0, size); });
  Allocate();
}

template <typename ValueType, std::size_t Order>
typename DenseTensor<ValueType, Order>::value_type& DenseTensor<ValueType, Order>::operator[](
    const pos_type& index) {
  const auto linear_index = internal::IndexHelper<Order, Order - 1>::Run(index, dimensions_);
  DCHECK_LT(linear_index, data_.size());
  return data_[linear_index];
}

template <typename ValueType, std::size_t Order>
typename DenseTensor<ValueType, Order>::value_type DenseTensor<ValueType, Order>::ComputeNorm()
    const {
  const auto squared_sum = std::accumulate(
      data_.cbegin(), data_.cend(), value_type(0),
      [](const value_type& lhs, const value_type& rhs) { return lhs + std::pow(rhs, 2); });
  return std::sqrt(squared_sum);
}

template <typename ValueType, std::size_t Order>
const typename DenseTensor<ValueType, Order>::value_type& DenseTensor<ValueType, Order>::operator[](
    const pos_type& index) const {
  const auto linear_index = internal::IndexHelper<Order, Order - 1>::Run(index, dimensions_);
  DCHECK_LT(linear_index, data_.size());
  return data_[linear_index];
}

template <typename ValueType, std::size_t Order>
typename DenseTensor<ValueType, Order>::pos_type DenseTensor<ValueType, Order>::dimensions() const {
  return dimensions_;
}

template <typename ValueType, std::size_t Order>
typename DenseTensor<ValueType, Order>::size_type DenseTensor<ValueType, Order>::size() const {
  return data_.size();
}

template <typename ValueType, std::size_t Order>
void DenseTensor<ValueType, Order>::Allocate() {
  const auto total_size = std::accumulate(dimensions_.cbegin(), dimensions_.cend(), 1,
                                          internal::MultiplyOrDie<std::uint64_t>);
  data_.resize(total_size);
}

#endif  // DENSE_TENSOR_INL_H_