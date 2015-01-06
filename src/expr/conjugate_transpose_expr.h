#ifndef EXPR_CONJUGATE_TRANSPOSE_EXPR_H_
#define EXPR_CONJUGATE_TRANSPOSE_EXPR_H_

#include <complex>
#include <type_traits>

#include "expr/matrix_expr.h"
#include "tensor/storage_mode.h"

namespace expr {
namespace {

template <typename T>
typename std::enable_if<std::is_same<std::complex<float>, T>::value ||
                            std::is_same<std::complex<double>, T>::value ||
                            std::is_same<std::complex<long double>, T>::value,
                        T>::type
Conjugate(const T& value) {
  return std::conj(value);
}

template <typename T>
typename std::enable_if<std::is_floating_point<T>::value, T>::type Conjugate(
    const T& value) {
  return value;
}

}  // namespace

template <typename ExprType, typename ValueType>
class ConjugateTransposeExpr
    : public MatrixExpr<ConjugateTransposeExpr<ExprType, ValueType>,
                        ValueType> {
 public:
  using base_type =
      MatrixExpr<ConjugateTransposeExpr<ExprType, ValueType>, ValueType>;
  using size_type = typename base_type::size_type;
  using pos_type = typename base_type::pos_type;
  using value_type = typename base_type::value_type;

  // Create a scalar product expression representing the product of a scalar and
  // a tensor.
  ConjugateTransposeExpr(const ExprType& expr) : expr_(expr) {}

  pos_type dimensions() const override {
    return pos_type { expr_.dimensions()[1], expr_.dimensions()[0] };
  }

  size_type size() const override {
    return expr_.size();
  }

  const value_type Get(const pos_type& index) const override {
    return Conjugate<value_type>(expr_.Get(pos_type{index[1], index[0]}));
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

} // namespace expr

template <typename ExprType, typename ValueType>
const expr::ConjugateTransposeExpr<ExprType, ValueType> operator~(
    const expr::MatrixExpr<ExprType, ValueType>& expr) {
  return expr::ConjugateTransposeExpr<ExprType, ValueType>(expr);
}

#endif  // EXPR_CONJUGATE_TRANSPOSE_EXPR_H_
