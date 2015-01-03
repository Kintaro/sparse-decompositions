#ifndef EXPR_TENSOR_EXPR_H_
#define EXPR_TENSOR_EXPR_H_

#include <array>
#include <cstddef>

namespace expr {

template <typename ExprType, typename ValueType, std::size_t Order>
class TensorExpr {
 public:
  using size_type = std::size_t;
  using pos_type = std::array<size_type, Order>;
  using value_type = ValueType;

  // Returns the dimensions of the tensor produced by this expression.
  virtual pos_type dimensions() const;

  // Returns the size of the tensor produced by this expression.
  virtual size_type size() const;

  // Retrieves the value at the designated index from within the tensor produced by this expression.
  virtual const value_type& operator[](const pos_type&) const;

  operator ExprType&() { return static_cast<ExprType&>(*this); }
  operator const ExprType&() { return static_cast<const ExprType&>(*this); }
};

}  // namespace expr

#include "tensor_expr-inl.h"

#endif  // EXPR_TENSOR_EXPR_H_
