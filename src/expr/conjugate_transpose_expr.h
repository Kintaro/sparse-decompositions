#ifndef EXPR_CONJUGATE_TRANSPOSE_EXPR_H_
#define EXPR_CONJUGATE_TRANSPOSE_EXPR_H_

#include <complex>
#include "expr/matrix_expr.h"
#include "tensor/storage_mode.h"

namespace expr {

template <typename ExprType, typename ValueType>
class ConjugateTransposeExpr
    : public MatrixExpr<ConjugateTransposeExpr<ExprType, ValueType>, ValueType> {
 public:
  using base_type = MatrixExpr<ConjugateTransposeExpr<ExprType, ValueType>, ValueType>;
  using size_type = typename base_type::size_type;
  using pos_type = typename base_type::pos_type;
  using value_type = typename base_type::value_type;

  // Create a scalar product expression representing the product of a scalar and a tensor.
  ConjugateTransposeExpr(const ExprType& expr);

  pos_type dimensions() const override {
    return expr_.dimensions();
  }

  size_type size() const override {
    return expr_.size();
  }

  const value_type Get(const pos_type& index) const override {
    return std::conj(expr_.Get(pos_type { index[1], index[0] }));
  }

  value_type abs() const override {
    CHECK(false) << "Not implemented yet!";
    return value_type{0};
  }

  tensor::StorageMode GetPreferredStorageMode() const override {
    return expr_.GetPreferredStorageMode();
  }

 private:
  const ExprType& expr_;
};

}  // namespace expr

#endif  // EXPR_SCALAR_PRODUCT_EXPR_H_
