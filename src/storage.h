#ifndef STORAGE_H_
#define STORAGE_H_

#include <algorithm>
#include <array>
#include <cstddef>
#include <memory>
#include <vector>

template <typename T, std::size_t P>
class Storage {
 public:
  using Index = std::array<std::size_t, P>;

  Storage() = delete;
  Storage(const Storage&) = delete;
  Storage(Storage&&) = delete;
  Storage& operator=(const Storage&) = delete;

  explicit Storage(const Index& size);
  explicit Storage(Index&& size);

  virtual const T& Get(const Index& index) const = 0;
  virtual void Set(const Index& index, const T& value) = 0;

  virtual std::unique_ptr<Storage> Copy() const = 0;

  inline std::size_t modes() const;

  inline std::size_t size(const std::size_t mode) const;
  inline const std::array<std::size_t, P>& size() const;

 private:
  constexpr static std::size_t modes_ = P;

  std::array<std::size_t, P> size_;
};

template <typename T, std::size_t P>
Storage<T, P>::Storage(const Storage<T, P>::Index& size)
    : size_(size) {}

template <typename T, std::size_t P>
Storage<T, P>::Storage(Storage<T, P>::Index&& size)
    : size_(std::move(size)) {}

template <typename T, std::size_t P>
std::size_t Storage<T, P>::modes() const {
  return modes_;
}

template <typename T, std::size_t P>
std::size_t Storage<T, P>::size(const std::size_t mode) const {
  return size_[mode];
}

template <typename T, std::size_t P>
const std::array<std::size_t, P>& Storage<T, P>::size() const {
  return size_;
}

#endif  // STORAGE_H_