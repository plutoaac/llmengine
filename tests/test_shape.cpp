#include <cassert>
#include <iostream>

#include "minillm/core/shape.h"

using namespace minillm;

void test_rank() {
    Shape s({2, 3, 4});
    assert(s.rank() == 3);
    assert(s.dim(0) == 2);
    assert(s.dim(1) == 3);
    assert(s.dim(2) == 4);
    std::cout << "  PASS test_rank\n";
}

void test_numel() {
    Shape s({2, 3, 4});
    auto n = s.numel();
    assert(n && *n == 24);
    std::cout << "  PASS test_numel\n";
}

void test_dynamic_dim() {
    Shape s({1, -1, 768});
    assert(s.has_dynamic_dim());
    assert(s.dim(1) == -1);
    auto n = s.numel();
    assert(!n);
    std::cout << "  PASS test_dynamic_dim\n";
}

void test_to_string() {
    Shape s({1, -1, 768});
    assert(s.to_string() == "[1, -1, 768]");
    std::cout << "  PASS test_to_string\n";
}

void test_equality() {
    Shape a({2, 3});
    Shape b({2, 3});
    Shape c({2, 4});
    assert(a == b);
    assert(!(a == c));
    std::cout << "  PASS test_equality\n";
}

void test_empty() {
    Shape s;
    assert(s.empty());
    assert(s.rank() == 0);
    std::cout << "  PASS test_empty\n";
}

int main() {
    std::cout << "test_shape:\n";
    test_rank();
    test_numel();
    test_dynamic_dim();
    test_to_string();
    test_equality();
    test_empty();
    std::cout << "All tests passed!\n";
    return 0;
}
