#ifndef TENSOR_TENSOR_H_
#define TENSOR_TENSOR_H_

#include <array>
#include <cstddef>
#include <algorithm>
#include <memory>
#include <glog/logging.h>
#include <limits>
#include <type_traits>
#include <vector>

#include "expr/tensor_expr.h"
#include "tensor/dense_storage.h"
#include "tensor/storage.h"
#include "tensor/storage_mode.h"
//#include "tensor/sparse_storage.h"

namespace tensor {

template <typename ValueType, std::size_t Order>
class Tensor : public expr::TensorExpr<Tensor<ValueType, Order>, ValueType, Order> {
 public:
  using type = Tensor<ValueType, Order>;
  using base_type = expr::TensorExpr<type, ValueType, Order>;
  using size_type = typename base_type::size_type;
  using pos_type = typename base_type::pos_type;
  using value_type = typename base_type::value_type;

  // Creates a tensor of the specified dimensions using the designated storage mode.
  Tensor(const pos_type& dimensions, const StorageMode& storage_mode) {
    switch (storage_mode) {
      case StorageMode::DENSE:
        storage_.reset(new DenseStorage<value_type, Order>(dimensions));
        break;
      case StorageMode::SPARSE:
        CHECK(false);  // Not implemented yet.
        // storage_.reset(new SparseStorage<value_type, Order>(dimensions));
        break;
    }
  }

  // Returns the designated tensor coefficient.
  const value_type Get(const pos_type& index) const override {
    DCHECK(storage_);
    return storage_->Get(index);
  }

  // Modifies the designated tensor coefficient.
  type& Set(const pos_type& index, const value_type& value) {
    DCHECK(storage_);
    storage_->Set(index, value);
    return *this;
  }

  // Returns the Frobenius norm of the tensor.
  value_type abs() const override {
    CHECK(false) << "Not implemented yet!";
    return value_type{0};
  }

  // Returns the dimensions of the tensor.
  pos_type dimensions() const override {
    DCHECK(storage_);
    return storage_->dimensions();
  }

  // Returns the size of the tensor.
  size_type size() const override {
    DCHECK(storage_);
    return storage_->size();
  }

  // Returns the storage mode of the tensor.
  StorageMode GetPreferredStorageMode() const override {
    DCHECK(storage_);
    return storage_->GetStorageMode();
  }

 private:
  std::unique_ptr<TensorStorage<value_type, Order>> storage_;
};

}  // namespace tensor

#endif  // TENSOR_TENSOR_H_
