// CP-ALS of order p; overview:
//
// (1) Initialize each A^(n).
// (2) Repeat until convergence
//     (2.1) For each i = 1..p
//           (2.1.1) Let C^(i) = adj(A^(i)) * A^(i)
//           (2.1.2) Let C = CoeffProd_{j=1..p, j != i} C^(j)
//           (2.1.3) Let A^(i) = X^(i) [KathriRao_{j=1..p, j != i} A^(j)] adj(C)
//
//
// Required structures:
//
// (1) Sparse Tensor
// (2) Dense Tensor
//
//
// Required operations:
//
// (1)

#include <cstdio>

#include "tensor.h"
#include "dense_tensor.h"
#include "storage.h"
#include "dense_storage.h"

int main(int argc, char** argv) {
  DenseStorage<float, 3> storage({{3, 3, 3}});

  for (std::size_t i = 0; i < storage.size(0); ++i) {
    for (std::size_t j = 0; j < storage.size(1); ++j) {
      for (std::size_t k = 0; k < storage.size(2); ++k) {
        storage.Set({{i, j, k}}, i + j + k);
      }
    }
  }

  for (std::size_t i = 0; i < storage.size(0); ++i) {
    for (std::size_t j = 0; j < storage.size(1); ++j) {
      for (std::size_t k = 0; k < storage.size(2); ++k) {
        printf("%lu %lu %lu => %f\n", i, j, k, storage.Get({{i, j, k}}));
      }
    }
  }

  return 0;
}