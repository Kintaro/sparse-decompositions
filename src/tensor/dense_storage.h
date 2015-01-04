#ifndef TENSOR_DENSE_STORAGE_H_
#define TENSOR_DENSE_STORAGE_H_

#include <algorithm>
#include <array>
#include <complex>
#include <type_traits>

#include "tensor/storage.h"

namespace tensor {
namespace internal {

// Returns the product of two unsigned values. If the multiplication would create an overflow the
// function dies instead.
template <typename ValueType>
typename std::enable_if<std::is_integral<ValueType>::value && std::is_unsigned<ValueType>::value,
                        ValueType>::type
MultiplyOrDie(const ValueType& lhs, const ValueType& rhs) {
  if (lhs != ValueType(0)) {
    CHECK_GT(std::numeric_limits<ValueType>::max() / lhs, rhs);
  }
  return lhs * rhs;
}

// A recursive template functor to help with the linearization of a variable-length indices into
// sequential 1D storage.
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
class DenseStorage : public TensorStorage<ValueType, Order> {
 public:
  static_assert(Order > 0, "DenseStorage requires strictly positive value for Order.");
  static_assert(std::is_floating_point<ValueType>::value ||
                    std::is_same<std::complex<float>, ValueType>::value || 
                    std::is_same<std::complex<double>, ValueType>::value ||
                    std::is_same<std::complex<long double>, ValueType>::value,                    
                "DenseStorage requires floating point or complex types for ValueType.");

  using type = DenseStorage<ValueType, Order>;
  using base_type = TensorStorage<ValueType, Order>;
  using size_type = typename base_type::size_type;
  using value_type = typename base_type::value_type;
  using pos_type = typename base_type::pos_type;

  // Allocates dense storage sufficiently large for the specified dimensions. The size of each
  // dimension must be strictly positive, the total number of coefficients must not overflow and the
  // tensor storage must not have order zero.
  explicit DenseStorage(const pos_type& dimensions)
      : base_type(dimensions), dimensions_(dimensions) {
    Allocate();
  }

  // Always returns dense storage mode.
  typename base_type::StorageMode GetStorageMode() const override {
    return base_type::DENSE;
  }

  // Returns the designated tensor coefficient.
  const value_type Get(const pos_type& index) const override {
    const auto linear_index = internal::IndexHelper<Order, Order - 1>::Run(index, dimensions_);
    DCHECK_LT(linear_index, data_.size());
    return data_[linear_index];
  }

  // Modifies the designated tensor coefficient.
  type& Set(const pos_type& index, const value_type& value) override {
    const auto linear_index = internal::IndexHelper<Order, Order - 1>::Run(index, dimensions_);
    DCHECK_LT(linear_index, data_.size());
    data_[linear_index] = value;
    return *this;
  }

  // Returns the dimensions of the stored tensor.
  const pos_type& dimensions() const override {
    return dimensions_;
  }

  // Returns the total number of coefficients in the stored tensor.
  size_type size() const override {
    return data_.size();
  }

 private:
  void Allocate() {
    CHECK_EQ(data_.size(), 0);  // Assure that Allocate() is invoked at most once.
    const auto total_size = std::accumulate(dimensions_.cbegin(), dimensions_.cend(), 1,
                                            internal::MultiplyOrDie<std::size_t>);
    data_.resize(total_size);
  }

  const pos_type dimensions_;
  std::vector<value_type> data_;  

  DenseStorage() = delete;
  DenseStorage(const DenseStorage&) = delete;
  DenseStorage& operator=(const DenseStorage&) = delete;
};

}  // namespace tensor

#endif  // TENSOR_DENSE_STORAGE_H_