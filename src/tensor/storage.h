#ifndef TENSOR_STORAGE_H_
#define TENSOR_STORAGE_H_

#include <cstddef>
#include <array>
#include <complex>
#include <glog/logging.h>
#include <type_traits>

namespace tensor {

// This class defines capabilities of a tensor storage implementation.
template <typename ValueType, std::size_t Order>
class TensorStorage {
 public:
  static_assert(Order > 0, "TensorStorage requires strictly positive value for Order.");
  static_assert(std::is_floating_point<ValueType>::value ||
                    std::is_same<std::complex<float>, ValueType>::value || 
                    std::is_same<std::complex<double>, ValueType>::value ||
                    std::is_same<std::complex<long double>, ValueType>::value,                    
                "TensorStorage requires floating point or complex types for ValueType.");

  using type = TensorStorage<ValueType, Order>;
  using size_type = std::size_t;
  using value_type = ValueType;
  using pos_type = std::array<size_type, Order>;

  // Supported tensor storage modes.
  enum StorageMode {
    DENSE,
    SPARSE
  };

  // Creates storage sufficiently large for the specified dimensions. The size of each dimension
  // must be strictly positive, the total number of coefficients must not overflow and the tensor
  // storage must not have order zero.
  explicit TensorStorage(const pos_type& dimensions) {
    std::for_each(dimensions.cbegin(), dimensions.cend(), [](const size_type& size) { 
      CHECK_LT(0, size); 
    });
  }

  // Returns the storage mode.
  virtual StorageMode GetStorageMode() const = 0;

  // Returns the designated tensor coefficient.
  virtual const value_type Get(const pos_type& index) const = 0;

  // Modifies the designated tensor coefficient.
  virtual type& Set(const pos_type& index, const value_type& value) = 0;

  // Returns the dimensions of the stored tensor.
  virtual const pos_type& dimensions() const = 0;

  // Returns the total number of coefficients in the stored tensor.
  virtual size_type size() const = 0;

 private:
  TensorStorage() = delete;
  TensorStorage(const TensorStorage&) = delete;
  TensorStorage& operator=(const TensorStorage&) = delete;
};

}  // namespace tensor

#endif  // TENSOR_STORAGE_H_