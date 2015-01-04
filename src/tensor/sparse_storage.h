#ifndef TENSOR_SPARSE_STORAGE_H_
#define TENSOR_SPARSE_STORAGE_H_

#include <algorithm>
#include <array>
#include <complex>
#include <type_traits>

#include "tensor/storage.h"

namespace tensor {

template <typename ValueType, std::size_t Order>
class SparseStorage : public TensorStorage<ValueType, Order> {
 public:
  static_assert(Order > 0, "SparseStorage requires strictly positive value for Order.");
  static_assert(std::is_floating_point<ValueType>::value ||
                    std::is_same<std::complex, ValueType>::value,
                "SparseStorage requires floating point or complex types for ValueType.");

  using base_type = TensorStorage<ValueType, Order>;
  using size_type = typename base_type::size_type;
  using value_type = typename base_type::value_type;
  using pos_type = typename base_type::pos_type;

  // Allocates dense storage sufficiently large for the specified dimensions. The size of each
  // dimension must be strictly positive, the total number of coefficients must not overflow and the
  // tensor storage must not have order zero.
  explicit SparseStorage(const pos_type& dimensions)
      : TensorStorage(dimensions), dimensions_(dimensions) {
    Allocate();
  }

  // Returns the storage mode.
  StorageMode GetStorageMode() const override {
    return StorageMode::SPARSE;
  }

  // Returns a writable reference to the designated tensor coefficient.
  value_type& operator[](const pos_type& index) override {
    const auto linear_index = IndexHelper<Order, Order - 1>::Run(index, dimensions_);
    DCHECK_LT(linear_index, data_.size());
    return data_[linear_index];
  }

  // Returns a non-writable reference to the designated tensor coefficient.
  const value_type& operator[](const pos_type& index) const override {
    const auto linear_index = IndexHelper<Order, Order - 1>::Run(index, dimensions_);
    DCHECK_LT(linear_index, data_.size());
    return data_[linear_index];
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
    CHECK_EQ(data_.size(), 0);
    const auto total_size =
        std::accumulate(dimensions_.cbegin(), dimensions_.cend(), 1, MultiplyOrDie<std::size_t>);
    data_.resize(total_size);
  }

  const pos_type dimensions_;
  std::vector<value_type> data_;  

  DenseStorage() = delete;
  DenseStorage(const DenseStorage&) = delete;
  DenseStorage& operator=(const DenseStorage&) = delete;
};

}  // namespace tensor

#endif  // TENSOR_SPARSE_STORAGE_H_