#include <gtest/gtest.h>
#include <array>
#include <list>
#include <vector>

#include "expr/expr_util.h"

namespace test {
namespace {

// Some static containers to test against.
const std::array<int, 3> array{1, 2, 3};
const std::list<int> list{1, 2, 3};
const std::vector<int> vector{1, 2, 3};

}  // namespace

TEST(ExprUtilTest, ContainerEqualsShouldFailOnDifferentLength) {
  std::list<int> short_list{1, 2};
  EXPECT_FALSE(expr::util::container_equals(list, short_list));
  std::vector<int> short_vector{1, 2};
  EXPECT_FALSE(expr::util::container_equals(vector, short_vector));
}

TEST(ExprUtilTest, ContainerEqualsShouldFailOnDifferentElementOrder) {
  std::array<int, 3> permuted_array{1, 3, 2};
  EXPECT_FALSE(expr::util::container_equals(array, permuted_array));
  std::list<int> permuted_list{2, 1, 3};
  EXPECT_FALSE(expr::util::container_equals(list, permuted_list));
  std::vector<int> permuted_vector{3, 1, 2};
  EXPECT_FALSE(expr::util::container_equals(vector, permuted_vector));
}

TEST(ExprUtilTest, ContainerEqualsShouldSucceedOnEquivalentContainers) {
  std::array<int, 3> identical_array{1, 2, 3};
  EXPECT_TRUE(expr::util::container_equals(array, identical_array));
  std::list<int> identical_list{1, 2, 3};
  EXPECT_TRUE(expr::util::container_equals(list, identical_list));
  std::vector<int> identical_vector{1, 2, 3};
  EXPECT_TRUE(expr::util::container_equals(vector, identical_vector));
}

TEST(ExprUtilTest, ContainerEqualsShouldFailWithEmptyContainers) {
  EXPECT_FALSE(expr::util::container_equals(array, {}));
  EXPECT_FALSE(expr::util::container_equals(list, {}));
  EXPECT_FALSE(expr::util::container_equals(vector, {}));
}

TEST(ExprUtilTest, ContainerEqualsShouldSucceedWithEmptyContainers) {
  std::array<int, 0> empty_array;
  std::list<int> empty_list;
  std::vector<int> empty_vector;
  EXPECT_TRUE(expr::util::container_equals(empty_array, {}));
  EXPECT_TRUE(expr::util::container_equals(empty_list, {}));
  EXPECT_TRUE(expr::util::container_equals(empty_vector, {}));
}

}  // namespace test