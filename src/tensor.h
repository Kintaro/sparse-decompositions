#ifndef TENSOR_H_
#define TENSOR_H_

#include <cstdint>
#include <memory>

#include "storage.h"

template <typename T, std::size_t P>
class Tensor {
 public:
  using Index = typename Storage<T, P>::Index;

  virtual const T& Get(const Index& index) = 0;
  virtual void Set(const Index& index, const T& value) = 0;
};

#endif  // TENSOR_H_