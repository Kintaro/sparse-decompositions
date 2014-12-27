#ifndef DENSE_TENSOR_INL_H_
#define DENSE_TENSOR_INL_H_

namespace internal {

// Returns the product of two unsigned values. If the multiplication would create an overflow the
// function dies instead.
template <typename T>
typename std::enable_if<std::is_arithmetic<T>::value&& std::is_unsigned<T>::value, T>::type
MultiplyOrDie(const T lhs, const T rhs) {
  if (lhs != T(0)) {
    CHECK_GT(std::numeric_limits<T>::max() / lhs, rhs);
  }
  return lhs * rhs;
}

// A variadic functor template to help with the linearization of a variable-length index into a
// tensor.
template <std::size_t P, std::size_t N>
struct IndexHelper {
  constexpr static inline std::size_t Run(const std::array<std::uint64_t, P>& index,
                                          const std::array<std::uint64_t, P>& size) {
    return index[P - N - 1] + size[P - N - 1] * IndexHelper<P, N - 1>::Run(index, size);
  }
};
template <std::size_t P>
struct IndexHelper<P, 0> {
  constexpr static inline std::size_t Run(const std::array<std::uint64_t, P>& index,
                                          const std::array<std::uint64_t, P>& size) {
    return index[P - 1];
  }
};

}  // namespace internal

template <typename T, std::size_t P>
DenseTensor<T, P>::DenseTensor(const Index& dimensions)
    : dimensions_(dimensions) {
  std::for_each(dimensions_.cbegin(), dimensions_.cend(),
                [](const Size size) { CHECK_LT(0, size); });
  Allocate();
}

template <typename T, std::size_t P>
const T& DenseTensor<T, P>::Get(const Index& index) const {
  const auto linear_index = internal::IndexHelper<P, P - 1>::Run(index, dimensions_);
  DCHECK_LT(linear_index, data_.size());
  return data_[linear_index];
}

template <typename T, std::size_t P>
typename DenseTensor<T, P>::Self& DenseTensor<T, P>::Set(const Index& index, const T value) {
  operator[](index) = value;
  return *this;
}

template <typename T, std::size_t P>
T& DenseTensor<T, P>::operator[](const Index& index) {
  const auto linear_index = internal::IndexHelper<P, P - 1>::Run(index, dimensions_);
  DCHECK_LT(linear_index, data_.size());
  return data_[linear_index];
}

template <typename T, std::size_t P>
typename std::enable_if<std::is_arithmetic<T>::value, T>::type DenseTensor<T, P>::ComputeNorm()
    const {
  const auto squared_sum =
      std::accumulate(data_.cbegin(), data_.cend(), T(0),
                      [](const T& lhs, const T& rhs) { return lhs + std::pow(rhs, 2); });
  return std::sqrt(squared_sum);
}

template <typename T, std::size_t P>
const T& DenseTensor<T, P>::operator[](const Index& index) const {
  return Get(index);
}

template <typename T, std::size_t P>
typename DenseTensor<T, P>::Index DenseTensor<T, P>::dimensions() const {
  return dimensions_;
}

template <typename T, std::size_t P>
typename DenseTensor<T, P>::Size DenseTensor<T, P>::size() const {
  return data_.size();
}

template <typename T, std::size_t P>
void DenseTensor<T, P>::Allocate() {
  const auto total_size = std::accumulate(dimensions_.cbegin(), dimensions_.cend(), 1,
                                          internal::MultiplyOrDie<std::uint64_t>);
  data_.resize(total_size);
}

#endif  // DENSE_TENSOR_INL_H_