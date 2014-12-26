#ifndef DENSE_STORAGE_H_
#define DENSE_STORAGE_H_

#include <algorithm>
#include <array>
#include <cstddef>
#include <memory>
#include <vector>

#include "storage.h"

namespace {

template <size_t P, std::size_t N>
struct DenseIndexLinearizer {
  constexpr static inline std::size_t Linearize(
      const std::array<std::size_t, P>& index,
      const std::array<std::size_t, P>& size) {
    return std::get<N>(index) +
           std::get<N>(size) *
               DenseIndexLinearizer<P, N - 1>::Linearize(index, size);
  }
};

template <size_t P>
struct DenseIndexLinearizer<P, 0> {
  constexpr static inline std::size_t Linearize(
      const std::array<std::size_t, P>& index,
      const std::array<std::size_t, P>& size) {
    return std::get<P - 1>(index);
  }
};

template <size_t N>
struct DenseIndexLinearizer<0, N> {};

}  // namespace

template <typename T, std::size_t P>
class DenseStorage : public Storage<T, P> {
 public:
  using Index = typename Storage<T, P>::Index;

  DenseStorage() = delete;
  DenseStorage(const DenseStorage&) = delete;
  DenseStorage(DenseStorage&&) = delete;
  DenseStorage& operator=(const DenseStorage&) = delete;
  DenseStorage& operator=(DenseStorage&&) = delete;

  inline explicit DenseStorage(const Index& size);

  const T& Get(const Index& index) const override;
  void Set(const Index& index, const T& value) override;

  std::unique_ptr<Storage<T, P>> Copy() const override;

 private:
  std::vector<T> data_;
};

template <typename T, std::size_t P>
DenseStorage<T, P>::DenseStorage(const DenseStorage<T, P>::Index& size)
    : Storage<T, P>(size) {}

template <typename T, std::size_t P>
std::unique_ptr<Storage<T, P>> DenseStorage<T, P>::Copy() const {
  DenseStorage* copy = new DenseStorage<T, P>(this->size());
  std::copy(data_.cbegin(), data_.cend(), copy->data_.begin());
  return std::unique_ptr<Storage<T, P>>(copy);
}

template <typename T, std::size_t P>
const T& DenseStorage<T, P>::Get(const DenseStorage<T, P>::Index& index) const {
  std::size_t linearized_index =
      DenseIndexLinearizer<P, P - 1>::Linearize(index, this->size());
  return data_[linearized_index];
}

template <typename T, std::size_t P>
void DenseStorage<T, P>::Set(const DenseStorage<T, P>::Index& index,
                             const T& value) {
  std::size_t linearized_index =
      DenseIndexLinearizer<P, P - 1>::Linearize(index, this->size());
  data_[linearized_index] = value;
}

#endif  // DENSE_STORAGE_H_