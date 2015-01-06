#ifndef TENSOR_DENSE_STORAGE_H_
#define TENSOR_DENSE_STORAGE_H_

#include <algorithm>
#include <array>
#include <complex>
#include <type_traits>

#include "tensor/storage.h"
#include "tensor/storage_mode.h"
#include "tensor/storage_util.h"

namespace tensor {

template <typename ValueType, std::size_t Order>
class DenseStorage : public TensorStorage<ValueType, Order> {
 public:
  static_assert(Order > 0,
                "DenseStorage requires strictly positive value for Order.");
  static_assert(
      std::is_floating_point<ValueType>::value ||
          std::is_same<std::complex<float>, ValueType>::value ||
          std::is_same<std::complex<double>, ValueType>::value ||
          std::is_same<std::complex<long double>, ValueType>::value,
      "DenseStorage requires floating point or complex types for ValueType.");

  using type = DenseStorage<ValueType, Order>;
  using base_type = TensorStorage<ValueType, Order>;
  using size_type = typename base_type::size_type;
  using value_type = typename base_type::value_type;
  using pos_type = typename base_type::pos_type;

  // Allocates dense storage sufficiently large for the specified dimensions.
  // The size of each dimension must be strictly positive, the total number of
  // coefficients must not overflow and the tensor storage must not have order
  // zero.
  explicit DenseStorage(const pos_type& dimensions)
      : base_type(dimensions), dimensions_(dimensions) {
    Allocate();
  }

  // Always returns dense storage mode.
  StorageMode GetStorageMode() const override { return StorageMode::DENSE; }

  // Returns the designated tensor coefficient.
  const value_type Get(const pos_type& index) const override {
    const auto linear_index =
        IndexHelper<Order, Order - 1>::Run(index, dimensions_);
    DCHECK_LT(linear_index, data_.size());
    return data_[linear_index];
  }

  // Modifies the designated tensor coefficient.
  type& Set(const pos_type& index, const value_type& value) override {
    const auto linear_index =
        IndexHelper<Order, Order - 1>::Run(index, dimensions_);
    DCHECK_LT(linear_index, data_.size());
    data_[linear_index] = value;
    return *this;
  }

  // Returns the dimensions of the stored tensor.
  const pos_type& dimensions() const override { return dimensions_; }

  // Returns the total number of coefficients in the stored tensor.
  size_type size() const override { return data_.size(); }

 private:
  void Allocate() {
    CHECK_EQ(data_.size(),
             0);  // Assure that Allocate() is invoked at most once.
    const auto total_size =
        std::accumulate(dimensions_.cbegin(), dimensions_.cend(), 1,
                        MultiplyOrDie<std::size_t>);
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
