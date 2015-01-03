#ifndef EXPR_EXPR_UTIL_H_
#define EXPR_EXPR_UTIL_H_

namespace expr {
namespace util {

// Returns whether the content of two containers is identical. Containers of different size are
// deemed non-identical.
template <typename ContainerType>
bool container_equals(const ContainerType& a, const ContainerType& b);

}  // namespace util
}  // namespace expr

#include "expr/expr_util-inl.h"

#endif  // EXPR_EXPR_UTIL_H_