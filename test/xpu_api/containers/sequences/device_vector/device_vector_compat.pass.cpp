// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) UXL Foundation Contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "support/test_config.h"

#include _PSTL_TEST_HEADER(execution)
#include _PSTL_TEST_HEADER(algorithm)
#include _PSTL_TEST_HEADER(numeric)
#include <oneapi/dpl/compat>

#include "support/utils.h"

#include <vector>
#include <numeric>

#if TEST_DPCPP_BACKEND_PRESENT

namespace compat = oneapi::dpl::experimental::compat;
namespace dpl_experimental = oneapi::dpl::experimental;

class KernelRawPointer;
class KernelVecSort;
class KernelVecTransform;
class KernelVecReduce;

// =====================================================================
// device_pointer tests
// =====================================================================

bool
test_device_pointer_basic()
{
    sycl::queue q = TestUtils::get_test_queue();
    compat::device_vector<int> d({10, 20, 30}, q);

    auto ptr = d.data();
    EXPECT_TRUE(ptr.get() != nullptr, "device_pointer: get() should not be null");
    EXPECT_TRUE(ptr.get() == d.begin().get(), "device_pointer: data().get() should equal begin().get()");
    return true;
}

bool
test_device_pointer_arithmetic()
{
    sycl::queue q = TestUtils::get_test_queue();
    compat::device_vector<int> d({1, 2, 3, 4, 5}, q);

    auto it = d.begin();
    auto it2 = it + 3;
    EXPECT_TRUE(it2 - it == 3, "device_pointer arithmetic: difference wrong");

    ++it;
    EXPECT_TRUE(it - d.begin() == 1, "device_pointer: prefix increment wrong");

    it += 2;
    EXPECT_TRUE(it - d.begin() == 3, "device_pointer: += wrong");

    --it;
    EXPECT_TRUE(it - d.begin() == 2, "device_pointer: decrement wrong");

    auto it3 = it - 1;
    EXPECT_TRUE(it3 - d.begin() == 1, "device_pointer: subtract wrong");
    return true;
}

bool
test_device_pointer_comparison()
{
    sycl::queue q = TestUtils::get_test_queue();
    compat::device_vector<int> d({1, 2, 3}, q);

    EXPECT_TRUE(d.begin() == d.begin(), "device_pointer: begin == begin");
    EXPECT_TRUE(d.begin() != d.end(), "device_pointer: begin != end");
    EXPECT_TRUE(d.begin() < d.end(), "device_pointer: begin < end");
    EXPECT_TRUE(d.end() > d.begin(), "device_pointer: end > begin");
    EXPECT_TRUE(d.begin() <= d.begin(), "device_pointer: begin <= begin");
    EXPECT_TRUE(d.end() >= d.end(), "device_pointer: end >= end");
    return true;
}

bool
test_raw_pointer_cast()
{
    sycl::queue q = TestUtils::get_test_queue();
    compat::device_vector<int> d({1, 2, 3}, q);

    int* raw = compat::raw_pointer_cast(d.data());
    EXPECT_TRUE(raw == d.data().get(), "raw_pointer_cast: should match get()");
    EXPECT_TRUE(raw != nullptr, "raw_pointer_cast: should not be null");
    return true;
}

bool
test_device_pointer_cast()
{
    sycl::queue q = TestUtils::get_test_queue();
    compat::device_vector<int> d({1, 2, 3}, q);

    int* raw = d.data().get();
    auto dp = compat::device_pointer_cast(raw);
    EXPECT_TRUE(dp.get() == raw, "device_pointer_cast: round-trip failed");
    return true;
}

// =====================================================================
// device_reference tests
// =====================================================================

bool
test_device_reference_read()
{
    sycl::queue q = TestUtils::get_test_queue();
    compat::device_vector<int> d({10, 20, 30}, q);

    int val = d[0];
    EXPECT_TRUE(val == 10, "device_reference read [0]: wrong value");
    int val2 = d[2];
    EXPECT_TRUE(val2 == 30, "device_reference read [2]: wrong value");
    return true;
}

bool
test_device_reference_write()
{
    sycl::queue q = TestUtils::get_test_queue();
    compat::device_vector<int> d({10, 20, 30}, q);

    d[1] = 99;
    int val = d[1];
    EXPECT_TRUE(val == 99, "device_reference write: wrong value after assignment");
    return true;
}

bool
test_device_reference_compound_assign()
{
    sycl::queue q = TestUtils::get_test_queue();
    compat::device_vector<int> d({10, 20, 30}, q);

    d[0] += 5;
    EXPECT_TRUE(static_cast<int>(d[0]) == 15, "+=: wrong result");

    d[1] -= 3;
    EXPECT_TRUE(static_cast<int>(d[1]) == 17, "-=: wrong result");

    d[2] *= 2;
    EXPECT_TRUE(static_cast<int>(d[2]) == 60, "*=: wrong result");
    return true;
}

bool
test_device_reference_increment_decrement()
{
    sycl::queue q = TestUtils::get_test_queue();
    compat::device_vector<int> d({10, 20}, q);

    ++d[0];
    EXPECT_TRUE(static_cast<int>(d[0]) == 11, "prefix ++: wrong result");

    d[0]++;
    EXPECT_TRUE(static_cast<int>(d[0]) == 12, "postfix ++: wrong result");

    --d[1];
    EXPECT_TRUE(static_cast<int>(d[1]) == 19, "prefix --: wrong result");

    d[1]--;
    EXPECT_TRUE(static_cast<int>(d[1]) == 18, "postfix --: wrong result");
    return true;
}

bool
test_device_reference_swap()
{
    sycl::queue q = TestUtils::get_test_queue();
    compat::device_vector<int> d({10, 20}, q);

    swap(d[0], d[1]);
    EXPECT_TRUE(static_cast<int>(d[0]) == 20, "swap: [0] should be 20");
    EXPECT_TRUE(static_cast<int>(d[1]) == 10, "swap: [1] should be 10");
    return true;
}

bool
test_front_back()
{
    sycl::queue q = TestUtils::get_test_queue();
    compat::device_vector<int> d({1, 2, 3}, q);

    int f = d.front();
    int b = d.back();
    EXPECT_TRUE(f == 1, "front: wrong value");
    EXPECT_TRUE(b == 3, "back: wrong value");

    d.front() = 100;
    d.back() = 300;
    EXPECT_TRUE(static_cast<int>(d.front()) == 100, "front assignment: wrong value");
    EXPECT_TRUE(static_cast<int>(d.back()) == 300, "back assignment: wrong value");
    return true;
}

// =====================================================================
// Construction tests
// =====================================================================

bool
test_construct_empty_from_queue()
{
    sycl::queue q = TestUtils::get_test_queue();
    compat::device_vector<int> d(q);
    EXPECT_TRUE(d.size() == 0, "empty construction: wrong size");
    EXPECT_TRUE(d.empty(), "empty construction: should be empty");
    return true;
}

bool
test_construct_sized()
{
    sycl::queue q = TestUtils::get_test_queue();
    compat::device_vector<int> d(100, q);
    EXPECT_TRUE(d.size() == 100, "sized construction: wrong size");
    return true;
}

bool
test_construct_with_value()
{
    sycl::queue q = TestUtils::get_test_queue();
    compat::device_vector<int> d(50, 42, q);
    EXPECT_TRUE(d.size() == 50, "value construction: wrong size");
    std::vector<int> result = static_cast<std::vector<int>>(d);
    bool all_correct = true;
    for (auto& v : result)
        all_correct &= (v == 42);
    EXPECT_TRUE(all_correct, "value construction: not all elements match");
    return true;
}

bool
test_construct_no_init()
{
    sycl::queue q = TestUtils::get_test_queue();
    compat::device_vector<int> d(100, dpl_experimental::no_init, q);
    EXPECT_TRUE(d.size() == 100, "no_init construction: wrong size");
    return true;
}

bool
test_construct_from_host_vector()
{
    sycl::queue q = TestUtils::get_test_queue();
    std::vector<int> src = {1, 2, 3, 4, 5};
    compat::device_vector<int> d(src, q);
    std::vector<int> result = static_cast<std::vector<int>>(d);
    EXPECT_TRUE(result == src, "host vector construction: data mismatch");
    return true;
}

bool
test_construct_from_host_iterators()
{
    sycl::queue q = TestUtils::get_test_queue();
    std::vector<int> src(100);
    std::iota(src.begin(), src.end(), 0);
    compat::device_vector<int> d(src.begin(), src.end(), q);
    std::vector<int> result = static_cast<std::vector<int>>(d);
    EXPECT_TRUE(result == src, "host iterator construction: data mismatch");
    return true;
}

bool
test_construct_from_initializer_list()
{
    sycl::queue q = TestUtils::get_test_queue();
    compat::device_vector<int> d({7, 8, 9}, q);
    EXPECT_TRUE(d.size() == 3, "initializer_list: wrong size");
    EXPECT_TRUE(static_cast<int>(d[0]) == 7, "initializer_list: [0] wrong");
    EXPECT_TRUE(static_cast<int>(d[1]) == 8, "initializer_list: [1] wrong");
    EXPECT_TRUE(static_cast<int>(d[2]) == 9, "initializer_list: [2] wrong");
    return true;
}

bool
test_construct_device_to_device()
{
    sycl::queue q = TestUtils::get_test_queue();
    compat::device_vector<int> d1({1, 2, 3}, q);
    compat::device_vector<int> d2(d1.begin(), d1.end(), q);
    std::vector<int> r1 = static_cast<std::vector<int>>(d1);
    std::vector<int> r2 = static_cast<std::vector<int>>(d2);
    EXPECT_TRUE(r1 == r2, "D2D construction: data mismatch");
    EXPECT_TRUE(d1.data().get() != d2.data().get(), "D2D: pointers should differ");
    return true;
}

bool
test_construct_from_ctx_dev()
{
    sycl::queue q = TestUtils::get_test_queue();
    sycl::context ctx = q.get_context();
    sycl::device dev = q.get_device();

    compat::device_vector<int> d1(ctx, dev);
    EXPECT_TRUE(d1.empty(), "ctx+dev empty: should be empty");

    compat::device_vector<int> d2(10, ctx, dev);
    EXPECT_TRUE(d2.size() == 10, "ctx+dev sized: wrong size");

    compat::device_vector<int> d3(10, 5, ctx, dev);
    EXPECT_TRUE(d3.size() == 10, "ctx+dev valued: wrong size");
    return true;
}

bool
test_copy_constructor()
{
    sycl::queue q = TestUtils::get_test_queue();
    compat::device_vector<int> d1({1, 2, 3}, q);
    compat::device_vector<int> d2(d1);
    std::vector<int> r1 = static_cast<std::vector<int>>(d1);
    std::vector<int> r2 = static_cast<std::vector<int>>(d2);
    EXPECT_TRUE(r1 == r2, "copy ctor: data mismatch");
    EXPECT_TRUE(d1.data().get() != d2.data().get(), "copy ctor: should be separate allocations");
    return true;
}

bool
test_move_constructor()
{
    sycl::queue q = TestUtils::get_test_queue();
    compat::device_vector<int> d1({1, 2, 3}, q);
    auto* old_raw = d1.data().get();
    compat::device_vector<int> d2(std::move(d1));
    EXPECT_TRUE(d2.data().get() == old_raw, "move ctor: should take ownership");
    EXPECT_TRUE(d1.empty(), "move ctor: source should be empty");
    return true;
}

bool
test_assign_from_host_vector()
{
    sycl::queue q = TestUtils::get_test_queue();
    compat::device_vector<int> d(q);
    std::vector<int> src = {10, 20, 30};
    d = src;
    std::vector<int> result = static_cast<std::vector<int>>(d);
    EXPECT_TRUE(result == src, "assign from host vector: data mismatch");
    return true;
}

// =====================================================================
// Capacity tests
// =====================================================================

bool
test_resize()
{
    sycl::queue q = TestUtils::get_test_queue();
    compat::device_vector<int> d({1, 2, 3}, q);

    d.resize(5);
    EXPECT_TRUE(d.size() == 5, "resize grow: wrong size");

    d.resize(2);
    EXPECT_TRUE(d.size() == 2, "resize shrink: wrong size");

    d.resize(4, 99);
    EXPECT_TRUE(d.size() == 4, "resize with value: wrong size");
    EXPECT_TRUE(static_cast<int>(d[2]) == 99, "resize with value: new element wrong");
    EXPECT_TRUE(static_cast<int>(d[3]) == 99, "resize with value: new element wrong");
    return true;
}

bool
test_reserve_and_capacity()
{
    sycl::queue q = TestUtils::get_test_queue();
    compat::device_vector<int> d({1, 2, 3}, q);
    d.reserve(1000);
    EXPECT_TRUE(d.size() == 3, "reserve: size should not change");
    EXPECT_TRUE(d.capacity() >= 1000, "reserve: capacity too small");
    return true;
}

bool
test_clear()
{
    sycl::queue q = TestUtils::get_test_queue();
    compat::device_vector<int> d({1, 2, 3}, q);
    d.clear();
    EXPECT_TRUE(d.empty(), "clear: should be empty");
    EXPECT_TRUE(d.size() == 0, "clear: size should be 0");
    return true;
}

bool
test_swap()
{
    sycl::queue q = TestUtils::get_test_queue();
    compat::device_vector<int> d1({1, 2, 3}, q);
    compat::device_vector<int> d2({4, 5}, q);

    d1.swap(d2);
    EXPECT_TRUE(d1.size() == 2, "swap: d1 wrong size");
    EXPECT_TRUE(d2.size() == 3, "swap: d2 wrong size");
    EXPECT_TRUE(static_cast<int>(d1[0]) == 4, "swap: d1[0] wrong");
    EXPECT_TRUE(static_cast<int>(d2[0]) == 1, "swap: d2[0] wrong");
    return true;
}

// =====================================================================
// Raw pointer usage in kernels
// =====================================================================

bool
test_raw_pointer_in_kernel()
{
    sycl::queue q = TestUtils::get_test_queue();
    compat::device_vector<int> d(10, 1, q);

    int* ptr = d.data().get();
    q.parallel_for<KernelRawPointer>(sycl::range<1>(d.size()), [=](sycl::id<1> i) { ptr[i] *= 3; }).wait();

    std::vector<int> result = static_cast<std::vector<int>>(d);
    bool all_correct = true;
    for (auto& v : result)
        all_correct &= (v == 3);
    EXPECT_TRUE(all_correct, "raw pointer kernel: wrong result");
    return true;
}

// =====================================================================
// Algorithm integration
// =====================================================================

bool
test_sort_via_iterators()
{
    sycl::queue q = TestUtils::get_test_queue();
    compat::device_vector<int> d({5, 3, 1, 4, 2}, q);

    auto policy = oneapi::dpl::execution::make_device_policy<KernelVecSort>(q);
    std::sort(policy, d.begin(), d.end());

    std::vector<int> result = static_cast<std::vector<int>>(d);
    std::vector<int> expected = {1, 2, 3, 4, 5};
    EXPECT_TRUE(result == expected, "sort: wrong result");
    return true;
}

bool
test_transform_via_iterators()
{
    sycl::queue q = TestUtils::get_test_queue();
    std::vector<int> src(50);
    std::iota(src.begin(), src.end(), 0);
    compat::device_vector<int> input(src, q);
    compat::device_vector<int> output(50, 0, q);

    auto policy = oneapi::dpl::execution::make_device_policy<KernelVecTransform>(q);
    std::transform(policy, input.begin(), input.end(), output.begin(), [](int x) { return x * 2; });

    std::vector<int> result = static_cast<std::vector<int>>(output);
    bool all_correct = true;
    for (std::size_t i = 0; i < result.size(); ++i)
        all_correct &= (result[i] == static_cast<int>(i * 2));
    EXPECT_TRUE(all_correct, "transform: wrong result");
    return true;
}

bool
test_reduce_via_iterators()
{
    sycl::queue q = TestUtils::get_test_queue();
    compat::device_vector<int> d(100, 1, q);

    auto policy = oneapi::dpl::execution::make_device_policy<KernelVecReduce>(q);
    int sum = std::reduce(policy, d.begin(), d.end(), 0);
    EXPECT_TRUE(sum == 100, "reduce: wrong result");
    return true;
}

// =====================================================================
// base() access for migration
// =====================================================================

bool
test_base_access()
{
    sycl::queue q = TestUtils::get_test_queue();
    compat::device_vector<int> d({1, 2, 3}, q);

    auto& arr = d.base();
    EXPECT_TRUE(arr.size() == 3, "base(): wrong size");
    EXPECT_TRUE(arr.read(0, q) == 1, "base() read: wrong value");

    arr.write(0, 99, q);
    EXPECT_TRUE(static_cast<int>(d[0]) == 99, "base() write reflected in device_vector");
    return true;
}

// =====================================================================
// Context / device
// =====================================================================

bool
test_context_device()
{
    sycl::queue q = TestUtils::get_test_queue();
    compat::device_vector<int> d(10, q);

    EXPECT_TRUE(d.get_context() == q.get_context(), "context mismatch");
    EXPECT_TRUE(d.get_device() == q.get_device(), "device mismatch");
    return true;
}

#endif // TEST_DPCPP_BACKEND_PRESENT

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    // device_pointer
    test_device_pointer_basic();
    test_device_pointer_arithmetic();
    test_device_pointer_comparison();
    test_raw_pointer_cast();
    test_device_pointer_cast();

    // device_reference
    test_device_reference_read();
    test_device_reference_write();
    test_device_reference_compound_assign();
    test_device_reference_increment_decrement();
    test_device_reference_swap();
    test_front_back();

    // Construction
    test_construct_empty_from_queue();
    test_construct_sized();
    test_construct_with_value();
    test_construct_no_init();
    test_construct_from_host_vector();
    test_construct_from_host_iterators();
    test_construct_from_initializer_list();
    test_construct_device_to_device();
    test_construct_from_ctx_dev();
    test_copy_constructor();
    test_move_constructor();
    test_assign_from_host_vector();

    // Capacity
    test_resize();
    test_reserve_and_capacity();
    test_clear();
    test_swap();

    // Kernel usage
    test_raw_pointer_in_kernel();

    // Algorithm integration
    test_sort_via_iterators();
    test_transform_via_iterators();
    test_reduce_via_iterators();

    // Migration
    test_base_access();

    // Context / device
    test_context_device();
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done();
}
