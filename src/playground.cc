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

#include <glog/logging.h>

#include <cstdio>

#include "tensor/dense_tensor.h"
// #include "expr/scalar_product_expr.h"

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);

  DenseTensor<float, 3> X({{2, 2, 2}});

  // X.Set({{0, 0, 0}}, 1.0f);

  // for (std::uint64_t i = 0; i < 2; ++i) {
  //   for (std::uint64_t j = 0; j < 2; ++j) {
  //     for (std::uint64_t k = 0; k < 2; ++k) {
  //       printf("%lu %lu %lu => %f\n", i, j, k, X.Get({{i, j, k}}));
  //     }
  //   }
  // }

  return 0;
}
