#ifndef DENSE_TENSOR_H_
#define DENSE_TENSOR_H_

#include <array>
#include <cstddef>
#include <algorithm>
#include <numeric>
#include <glog/logging.h>
#include <limits>
#include <type_traits>
#include <vector>

template <typename T, std::size_t P>
class DenseTensor {
 public:
  using Size = std::size_t;
  using Index = std::array<Size, P>;
  using Self = DenseTensor<T, P>;

  // Creates a tensor of the specified dimensions. Individual dimensions must be strictly positive
  // and the resulting tensor size must not overflow.
  explicit DenseTensor(const Index& dimensions);

  // Returns a non-writable reference to the designated tensor coefficient.
  inline const T& Get(const Index& index) const;

  // Modifies the designated tensor coefficient.
  inline Self& Set(const Index& index, const T value);

  // Returns a writable reference to the designated tensor coefficient.
  inline T& operator[](const Index& index);

  // Returns a non-writable reference to the designated tensor coefficient.
  inline const T& operator[](const Index& index) const;

  // Returns the Frobenius norm of the tensor.
  inline typename std::enable_if<std::is_arithmetic<T>::value, T>::type ComputeNorm() const;

  // Returns the dimensions of the tensor.
  inline Index dimensions() const;

  // Returns the size of the tensor.
  inline Size size() const;

 private:
  // Allocates storage for tensor coefficients.
  void Allocate();

  // Dimensions of the individual tensor modes.
  const Index dimensions_;

  // Coefficient storage.
  std::vector<T> data_;
};

#include "dense_tensor-inl.h"

#endif  // DENSE_TENSOR_H_
