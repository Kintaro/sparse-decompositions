#include "expr_util.h"

#define TEMPLATE template <typename FirstExprType, typename SecondExprType, typename ValueType>
#define CLASS InnerProductExpr<FirstExprType, SecondExprType, ValueType>

namespace expr {

TEMPLATE CLASS::InnerProductExpr(const FirstExprType& first_expr, const SecondExprType& second_expr)
    : first_expr_(first_expr), second_expr_(second_expr) {
  CHECK_EQ(1, first_expr_.dimensions().size());
  CHECK(util::container_equals(first_expr_.dimensions(), second_expr_.dimensions());
}

TEMPLATE typename CLASS::pos_type CLASS::dimensions() const override {
  return pos_type{1};
}

TEMPLATE typename CLASS::size_type CLASS::size() const override {
  return size_type{1};
}

TEMPLATE const typename CLASS::value_type CLASS::operator[](const pos_Type& i) const override {
  // TODO
}

TEMPLATE typename CLASS::value_type CLASS::abs() const override {
  // TODO
}

}  // namespace expr

#undef CLASS
#undef TEMPLATE
