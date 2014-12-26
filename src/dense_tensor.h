#ifndef DENSE_TENSOR_H_
#define DENSE_TENSOR_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <glog/logging.h>
#include <limits>
#include <type_traits>
#include <vector>

namespace {

// Returns the product of two unsigned values. If the multiplication would create an overflow the
// function dies instead.
template <typename T>
typename std::enable_if<std::is_arithmetic<T>::value&& std::is_unsigned<T>::value, T>::type
MultiplyOrDie(const T lhs, const T rhs) {
  CHECK_GT(std::numeric_limits<T>::max() / lhs, rhs);
  return lhs * rhs;
}

// TODO(mnett): Modify the order to maximize throughput for matrix operations [P=2].
template <std::size_t P, std::size_t N>
struct IndexHelper {
  constexpr static inline std::uint64_t LinearizeIndex(const std::array<std::uint64_t, P>& index,
                                                       const std::array<std::uint64_t, P>& size) {
    return index[P - N - 1] + size[P - N - 1] * IndexHelper<P, N - 1>::LinearizeIndex(index, size);
  }
};

template <std::size_t P>
struct IndexHelper<P, 0> {
  constexpr static inline std::uint64_t LinearizeIndex(const std::array<std::uint64_t, P>& index,
                                                       const std::array<std::uint64_t, P>& size) {
    return index[P - 1];
  }
};

}  // namespace

template <typename T, std::size_t P>
class DenseTensor {
 public:
  using Index = std::array<std::uint64_t, P>;
  using Self = DenseTensor<T, P>;

  explicit DenseTensor(const Index& size);

  // Returns the desired coefficient.
  inline const T& Get(const Index& index) const;

  // Modifies the desired coefficient.
  inline Self& Set(const Index& index, const T value);

  // Returns the size of the individual modes of the tensor.
  inline Index size() const { return size_; }

  // Returns the volume of the tensor.
  inline std::uint64_t volume() const { return data_.size(); }

 private:
  // Allocates tensor storage.
  void Allocate();

  // Sizes of individual tensor modes.
  const Index size_;

  // Coefficient storage.
  std::vector<T> data_;
};

template <typename T, std::size_t P>
DenseTensor<T, P>::DenseTensor(const Index& size)
    : size_(size) {
  for (auto mode = 0; mode < P; ++mode) {
    CHECK_LT(0, size_[mode]);
  }
  Allocate();
}

template <typename T, std::size_t P>
const T& DenseTensor<T, P>::Get(const Index& index) const {
  const std::uint64_t linear_index = IndexHelper<P, P - 1>::LinearizeIndex(index, size_);
  return data_[linear_index];
}

template <typename T, std::size_t P>
DenseTensor<T, P>& DenseTensor<T, P>::Set(const Index& index, const T value) {
  const std::uint64_t linear_index = IndexHelper<P, P - 1>::LinearizeIndex(index, size_);
  data_[linear_index] = value;
  return *this;
}

template <typename T, std::size_t P>
void DenseTensor<T, P>::Allocate() {
  std::uint64_t total_size = 1;
  for (const std::uint64_t size : size_) {
    total_size = MultiplyOrDie(total_size, size);
  }
  data_.resize(total_size);
}

#endif  // DENSE_TENSOR_H_