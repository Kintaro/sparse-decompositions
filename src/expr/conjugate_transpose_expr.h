#ifndef EXPR_CONJUGATE_TRANSPOSE_EXPR_H_
#define EXPR_CONJUGATE_TRANSPOSE_EXPR_H_

#include <complex>
#include "expr/matrix_expr.h"
#include "tensor/storage_mode.h"

namespace expr {

namespace internal {
  template <class T>
  typename std::enable_if<std::is_same<std::complex<float>, T>::value ||
      std::is_same<std::complex<double>, T>::value ||
      std::is_same<std::complex<long double>, T>::value, T>::type conjugate(T t) {
    return std::conj(t);
  }

  template <typename T>
  typename std::enable_if<std::is_floating_point<T>::value, T>::type conjugate(T t) {
    return t;
  }

  template <bool Conj, typename ValueType> struct Conjugator {
    static const ValueType apply(const ValueType& value);
  };

  template <typename ValueType> struct Conjugator<true, ValueType> {
    static const ValueType apply(const ValueType& value) { return conjugate(value); }
  };

  template <typename ValueType> struct Conjugator<false, ValueType> {
    static const ValueType apply(const ValueType& value) { return value; }
  };
} // namespace internal

template <typename ExprType, typename ValueType, typename Conj = internal::Conjugator<true, ValueType>>
class ConjugateTransposeExpr
    : public MatrixExpr<ConjugateTransposeExpr<ExprType, ValueType, Conj>, ValueType> {
 public:
  using base_type  = MatrixExpr<ConjugateTransposeExpr<ExprType, ValueType, Conj>, ValueType>;
  using size_type  = typename base_type::size_type;
  using pos_type   = typename base_type::pos_type;
  using value_type = typename base_type::value_type;

  // Create a scalar product expression representing the product of a scalar and a tensor.
  ConjugateTransposeExpr(const ExprType& expr) : expr_(expr) {}

  pos_type dimensions() const override {
    return expr_.dimensions();
  }

  size_type size() const override {
    return expr_.size();
  }

  const value_type Get(const pos_type& index) const override {
    return Conj::apply(expr_.Get(pos_type { index[1], index[0] }));
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
