#ifndef EXPR_TENSOR_EXPR_H_
#define EXPR_TENSOR_EXPR_H_

#include <array>
#include <cstddef>
#include <glog/logging.h>

#include "tensor/storage_mode.h"

namespace expr {

// TODO(mnett): Introduce storage types and add materialize() function.
template <typename ExprType, typename ValueType, std::size_t Order>
class TensorExpr {
 public:
  using size_type = std::size_t;
  using pos_type = std::array<size_type, Order>;
  using value_type = ValueType;

  // Returns the dimensions of the tensor produced by this expression.
  virtual pos_type dimensions() const {
    return static_cast<const ExprType&>(*this).dimensions();
  }

  // Returns the size of the tensor produced by this expression.
  virtual size_type size() const {
    return static_cast<const ExprType&>(*this).size();
  }

  // Retrieves the value at the designated index from within the tensor produced by this expression.
  virtual const value_type Get(const pos_type& index) const {
    return static_cast<const ExprType&>(*this).Get(index);
  }

  // Returns the Frobenius norm of the tensor produced by this expression.
  virtual value_type abs() const {
    CHECK(false) << "Not implemented yet!";
    return value_type{0};
  }

  // Returns the preferred storage mode for the tensor produced by this expression.
  virtual tensor::StorageMode GetPreferredStorageMode() const {
    return static_cast<const ExprType&>(*this).GetPreferredStorageMode();
  }

  operator ExprType&() { 
    return static_cast<ExprType&>(*this); 
  }
  
  operator const ExprType&() const { 
    return static_cast<const ExprType&>(*this); 
  }
};

}  // namespace expr

#endif  // EXPR_TENSOR_EXPR_H_
