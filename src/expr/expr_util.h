#ifndef EXPR_EXPR_UTIL_H_
#define EXPR_EXPR_UTIL_H_

namespace expr {
namespace util {

// Returns whether the content of two containers is identical. Containers of different size are
// deemed non-identical.
template <typename ContainerType>
bool container_equals(const ContainerType& a, const ContainerType& b) {
  auto iter_a = a.cbegin();
  auto iter_b = b.cbegin();
  while ((iter_a != a.cend()) && (iter_b != b.cend())) {
    if (*iter_a != *iter_b) {
      return false;
    }
    ++iter_a;
    ++iter_b;
  }
  if ((iter_a != a.cend()) || (iter_b != b.cend())) {
    return false;
  }
  return true;
}

}  // namespace util
}  // namespace expr

#endif  // EXPR_EXPR_UTIL_H_
