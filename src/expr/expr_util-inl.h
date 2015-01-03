namespace expr {
namespace util {

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
