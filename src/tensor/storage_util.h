#ifndef TENSOR_STORAGE_UTIL_H_
#define TENSOR_STORAGE_UTIL_H_

#include <array>
#include <glog/logging.h>
#include <type_traits>

namespace tensor {

// Returns the product of two unsigned values. If the multiplication would
// create an overflow the function dies instead.
template <typename ValueType>
typename std::enable_if<std::is_integral<ValueType>::value &&
                            std::is_unsigned<ValueType>::value,
                        ValueType>::type
MultiplyOrDie(const ValueType& lhs, const ValueType& rhs) {
  if (lhs != ValueType(0)) {
    CHECK_GT(std::numeric_limits<ValueType>::max() / lhs, rhs);
  }
  return lhs * rhs;
}

// A recursive template functor to help with the linearization of a
// variable-length indices into sequential one-dimensional storage.
template <std::size_t Order, std::size_t N>
struct IndexHelper {
  constexpr static inline std::size_t Run(
      const std::array<std::uint64_t, Order>& index,
      const std::array<std::uint64_t, Order>& size) {
    return index[Order - N - 1] +
           size[Order - N - 1] * IndexHelper<Order, N - 1>::Run(index, size);
  }
};
template <std::size_t Order>
struct IndexHelper<Order, 0> {
  constexpr static inline std::size_t Run(
      const std::array<std::uint64_t, Order>& index,
      const std::array<std::uint64_t, Order>& size) {
    return index[Order - 1];
  }
};

}  // namespace

#endif  // TENSOR_STORAGE_UTIL_H_
