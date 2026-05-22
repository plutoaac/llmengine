#include <cassert>
#include <iostream>

#include "minillm/core/tensor.h"

using namespace minillm;

void test_nbytes() {
    Tensor t("x", Shape({2, 3}), DType::Float32);
    auto nb = t.nbytes();
    assert(nb && *nb == 2 * 3 * 4);
    std::cout << "  PASS test_nbytes\n";
}

void test_allocate_cpu() {
    Tensor t("x", Shape({2, 3}), DType::Float32);
    assert(!t.is_allocated());
    auto st = t.allocate_cpu();
    assert(st.ok());
    assert(t.is_allocated());
    assert(t.data() != nullptr);
    std::cout << "  PASS test_allocate_cpu\n";
}

void test_dynamic_shape_allocate_fails() {
    Tensor t("x", Shape({1, -1, 768}), DType::Float32);
    auto st = t.allocate_cpu();
    assert(!st.ok());
    std::cout << "  PASS test_dynamic_shape_allocate_fails\n";
}

void test_bind_cpu_data_view() {
    alignas(64) std::byte storage[64];
    Tensor t("x", Shape({2, 3}), DType::Float32);
    auto st = t.bind_cpu_data(storage, sizeof(storage));
    assert(st.ok());
    assert(t.is_allocated());
    assert(t.data() == storage);

    float* data = reinterpret_cast<float*>(t.data());
    data[0] = 3.0f;
    assert(reinterpret_cast<float*>(storage)[0] == 3.0f);
    std::cout << "  PASS test_bind_cpu_data_view\n";
}

void test_unsupported_dtype_nbytes() {
    Tensor t("x", Shape({2, 3}), DType::Unknown);
    auto nb = t.nbytes();
    assert(!nb);
    std::cout << "  PASS test_unsupported_dtype_nbytes\n";
}

int main() {
    std::cout << "test_tensor:\n";
    test_nbytes();
    test_allocate_cpu();
    test_dynamic_shape_allocate_fails();
    test_bind_cpu_data_view();
    test_unsupported_dtype_nbytes();
    std::cout << "All tests passed!\n";
    return 0;
}
