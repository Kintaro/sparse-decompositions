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

#include "expr/tensor_expr.h"

template <typename ValueType, std::size_t Order>
class DenseTensor : public expr::TensorExpr<DenseTensor<ValueType, Order>, ValueType, Order> {
 public:
  using base_type = expr::TensorExpr<DenseTensor<ValueType, Order>, ValueType, Order>;
  using size_type = typename base_type::size_type;
  using pos_type = typename base_type::pos_type;
  using value_type = typename base_type::value_type;

  // Creates a tensor of the specified dimensions. Individual dimensions must be strictly positive
  // and the resulting tensor size must not overflow.
  explicit DenseTensor(const pos_type&);

  // Returns a writable reference to the designated tensor coefficient.
  value_type& operator[](const pos_type&);

  // Returns a non-writable reference to the designated tensor coefficient.
  const value_type& operator[](const pos_type&) const override;

  // Returns the Frobenius norm of the tensor.
  value_type ComputeNorm() const;

  // Returns the dimensions of the tensor.
  pos_type dimensions() const;

  // Returns the size of the tensor.
  size_type size() const;

 private:
  // Allocates storage for tensor coefficients.
  void Allocate();

  // Dimensions of the individual tensor modes.
  const pos_type dimensions_;

  // Coefficient storage.
  std::vector<value_type> data_;
};

#include "dense_tensor-inl.h"

#endif  // DENSE_TENSOR_H_
