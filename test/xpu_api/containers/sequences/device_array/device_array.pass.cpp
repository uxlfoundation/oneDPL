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

#if _ENABLE_RANGES_TESTING
#    include <oneapi/dpl/ranges>
#endif

#include "support/utils.h"

#include <vector>
#include <numeric>
#if _ENABLE_STD_RANGES_TESTING
#    include <ranges>
#    if TEST_CPP20_SPAN_PRESENT
#        include <span>
#    endif
#endif

#include <deque>
#include <list>

#if TEST_DPCPP_BACKEND_PRESENT

namespace dpl_experimental = oneapi::dpl::experimental;

struct point_t
{
    float x;
    float y;
    float z;
    int id;

    bool
    operator==(const point_t& o) const
    {
        return x == o.x && y == o.y && z == o.z && id == o.id;
    }
    bool
    operator!=(const point_t& o) const
    {
        return !(*this == o);
    }
};

class KernelDataPointer;
class KernelSpanInKernel;
class KernelSort;
class KernelTransformIn;
class KernelTransformOut;
class KernelReduce;
class KernelFill;
class KernelCopySrc;
class KernelCopyDst;
class KernelSpanSort;
class KernelSpanForEach;
class KernelSpanTransform;
class KernelStdRangesSort;
class KernelStdRangesForEach;
class KernelStdRangesTransform;
class KernelStdRangesSortLvalue;
class KernelStdRangesForEachLvalue;
class KernelStdRangesTransformLvalue;
class KernelStdRangesSortTake;
class KernelStdRangesForEachReverse;
class KernelStdRangesCountIfDrop;
class KernelStdSpanSort;
class KernelStdSpanForEach;
class KernelSpanConstInKernel;
class KernelStdSpanInKernel;
class KernelStdSpanConstInKernel;
class KernelStructScale;
class KernelStructTransform;

// =====================================================================
// Construction tests
// =====================================================================

bool
test_construct_uninitialized_from_queue()
{
    sycl::queue q = TestUtils::get_test_queue();
    dpl_experimental::device_array<float> d(1024, q);
    EXPECT_TRUE(d.size() == 1024, "uninitialized construction: wrong size");
    EXPECT_TRUE(!d.empty(), "uninitialized construction: should not be empty");
    EXPECT_TRUE(d.data() != nullptr, "uninitialized construction: null data pointer");
    return true;
}

bool
test_construct_uninitialized_from_ctx_dev()
{
    sycl::queue q = TestUtils::get_test_queue();
    sycl::context ctx = q.get_context();
    sycl::device dev = q.get_device();
    dpl_experimental::device_array<int> d(512, ctx, dev);
    EXPECT_TRUE(d.size() == 512, "ctx+dev construction: wrong size");
    EXPECT_TRUE(d.data() != nullptr, "ctx+dev construction: null data pointer");
    return true;
}

bool
test_construct_with_value()
{
    sycl::queue q = TestUtils::get_test_queue();
    dpl_experimental::device_array<int> d(100, 42, q);
    std::vector<int> result = d.to_vector(q);
    EXPECT_TRUE(result.size() == 100, "value construction: wrong size");
    bool all_correct = true;
    for (auto& v : result)
        all_correct &= (v == 42);
    EXPECT_TRUE(all_correct, "value construction: not all elements match");
    return true;
}

bool
test_construct_from_iterators()
{
    sycl::queue q = TestUtils::get_test_queue();
    std::vector<int> src(256);
    std::iota(src.begin(), src.end(), 0);
    dpl_experimental::device_array<int> d(src.begin(), src.end(), q);
    std::vector<int> result = d.to_vector(q);
    EXPECT_TRUE(result == src, "iterator construction: data mismatch");
    return true;
}

bool
test_construct_from_initializer_list()
{
    sycl::queue q = TestUtils::get_test_queue();
    dpl_experimental::device_array<int> d({1, 2, 3, 4, 5}, q);
    std::vector<int> result = d.to_vector(q);
    std::vector<int> expected = {1, 2, 3, 4, 5};
    EXPECT_TRUE(result == expected, "initializer_list construction: data mismatch");
    return true;
}

bool
test_construct_from_vector()
{
    sycl::queue q = TestUtils::get_test_queue();
    std::vector<float> src = {1.0f, 2.0f, 3.0f};
    dpl_experimental::device_array<float> d(src, q);
    std::vector<float> result = d.to_vector(q);
    EXPECT_TRUE(result == src, "vector construction: data mismatch");
    return true;
}

bool
test_copy_constructor()
{
    sycl::queue q = TestUtils::get_test_queue();
    std::vector<int> src = {10, 20, 30, 40};
    dpl_experimental::device_array<int> d1(src, q);
    dpl_experimental::device_array<int> d2(d1);
    std::vector<int> r1 = d1.to_vector(q);
    std::vector<int> r2 = d2.to_vector(q);
    EXPECT_TRUE(r1 == src, "copy ctor: original changed");
    EXPECT_TRUE(r2 == src, "copy ctor: copy doesn't match");
    EXPECT_TRUE(d1.data() != d2.data(), "copy ctor: pointers should differ");
    return true;
}

bool
test_move_constructor()
{
    sycl::queue q = TestUtils::get_test_queue();
    std::vector<int> src = {10, 20, 30};
    dpl_experimental::device_array<int> d1(src, q);
    auto* old_ptr = d1.data();
    dpl_experimental::device_array<int> d2(std::move(d1));
    EXPECT_TRUE(d2.data() == old_ptr, "move ctor: should take ownership of pointer");
    EXPECT_TRUE(d1.empty(), "move ctor: source should be empty");
    std::vector<int> r2 = d2.to_vector(q);
    EXPECT_TRUE(r2 == src, "move ctor: data mismatch");
    return true;
}

bool
test_copy_assignment()
{
    sycl::queue q = TestUtils::get_test_queue();
    dpl_experimental::device_array<int> d1({1, 2, 3}, q);
    dpl_experimental::device_array<int> d2({9, 8, 7, 6}, q);
    d2 = d1;
    std::vector<int> r1 = d1.to_vector(q);
    std::vector<int> r2 = d2.to_vector(q);
    std::vector<int> expected = {1, 2, 3};
    EXPECT_TRUE(r1 == expected, "copy assign: original changed");
    EXPECT_TRUE(r2 == expected, "copy assign: copy doesn't match");
    return true;
}

bool
test_move_assignment()
{
    sycl::queue q = TestUtils::get_test_queue();
    dpl_experimental::device_array<int> d1({1, 2, 3}, q);
    dpl_experimental::device_array<int> d2({9, 8}, q);
    auto* old_ptr = d1.data();
    d2 = std::move(d1);
    EXPECT_TRUE(d2.data() == old_ptr, "move assign: should take ownership");
    EXPECT_TRUE(d1.empty(), "move assign: source should be empty");
    std::vector<int> r2 = d2.to_vector(q);
    std::vector<int> expected = {1, 2, 3};
    EXPECT_TRUE(r2 == expected, "move assign: data mismatch");
    return true;
}

// =====================================================================
// Host-device transfer tests
// =====================================================================

bool
test_to_vector()
{
    sycl::queue q = TestUtils::get_test_queue();
    std::vector<int> src = {5, 10, 15, 20, 25};
    dpl_experimental::device_array<int> d(src, q);

    std::vector<int> r1 = d.to_vector(q);
    EXPECT_TRUE(r1 == src, "to_vector(q): data mismatch");

    std::vector<int> r2 = d.to_vector();
    EXPECT_TRUE(r2 == src, "to_vector(): data mismatch");
    return true;
}

bool
test_assign_from_pointers()
{
    sycl::queue q = TestUtils::get_test_queue();
    dpl_experimental::device_array<int> d(3, 0, q);
    std::vector<int> src = {100, 200, 300, 400, 500};
    d.assign(src.data(), src.data() + src.size(), q);
    std::vector<int> result = d.to_vector(q);
    EXPECT_TRUE(result == src, "assign from pointers: data mismatch");
    EXPECT_TRUE(d.size() == 5, "assign from pointers: wrong size after resize");
    return true;
}

bool
test_assign_from_vector()
{
    sycl::queue q = TestUtils::get_test_queue();
    dpl_experimental::device_array<int> d(2, 0, q);
    std::vector<int> src = {7, 8, 9};
    d.assign(src, q);
    std::vector<int> result = d.to_vector(q);
    EXPECT_TRUE(result == src, "assign from vector: data mismatch");
    return true;
}

bool
test_read_write()
{
    sycl::queue q = TestUtils::get_test_queue();
    dpl_experimental::device_array<int> d({10, 20, 30}, q);

    EXPECT_TRUE(d.read(0, q) == 10, "read(0): wrong value");
    EXPECT_TRUE(d.read(1, q) == 20, "read(1): wrong value");
    EXPECT_TRUE(d.read(2, q) == 30, "read(2): wrong value");

    d.write(1, 99, q);
    EXPECT_TRUE(d.read(1, q) == 99, "write then read: wrong value");

    EXPECT_TRUE(d.read(0) == 10, "read(0) no queue: wrong value");
    d.write(2, 77);
    EXPECT_TRUE(d.read(2) == 77, "write(2) no queue then read: wrong value");
    return true;
}

bool
test_async_read_write()
{
    sycl::queue q = TestUtils::get_test_queue();
    dpl_experimental::device_array<int> d({10, 20, 30}, q);

    int out = 0;
    sycl::event e = d.async_read(1, out, q);
    e.wait();
    EXPECT_TRUE(out == 20, "async_read: wrong value");

    sycl::event e2 = d.async_write(1, 55, q);
    e2.wait();
    int out2 = 0;
    sycl::event e3 = d.async_read(1, out2, q, {e2});
    e3.wait();
    EXPECT_TRUE(out2 == 55, "async_write then async_read: wrong value");
    return true;
}

bool
test_async_bulk_transfer()
{
    sycl::queue q = TestUtils::get_test_queue();
    std::vector<int> src = {1, 2, 3, 4, 5};
    dpl_experimental::device_array<int> d(5, q);

    sycl::event e_up = d.async_assign(src.data(), src.data() + src.size(), q);

    std::vector<int> dst(5);
    sycl::event e_down = d.async_to_vector(dst, q, {e_up});
    e_down.wait();

    EXPECT_TRUE(dst == src, "async bulk transfer: data mismatch");
    return true;
}

// =====================================================================
// Iteration and data access tests
// =====================================================================

bool
test_iterators_with_algorithm()
{
    sycl::queue q = TestUtils::get_test_queue();
    std::vector<int> src = {5, 3, 1, 4, 2};
    dpl_experimental::device_array<int> d(src, q);

    auto policy = oneapi::dpl::execution::make_device_policy<KernelSort>(q);
    std::sort(policy, d.begin(), d.end());

    std::vector<int> result = d.to_vector(q);
    std::vector<int> expected = {1, 2, 3, 4, 5};
    EXPECT_TRUE(result == expected, "sort via iterators: wrong result");
    return true;
}

bool
test_data_pointer_in_kernel()
{
    sycl::queue q = TestUtils::get_test_queue();
    dpl_experimental::device_array<int> d(10, 1, q);

    int* ptr = d.data();
    q.parallel_for<KernelDataPointer>(sycl::range<1>(d.size()), [=](sycl::id<1> i) { ptr[i] *= 2; }).wait();

    std::vector<int> result = d.to_vector(q);
    bool all_correct = true;
    for (auto& v : result)
        all_correct &= (v == 2);
    EXPECT_TRUE(all_correct, "kernel via data pointer: wrong result");
    return true;
}

bool
test_const_iterators()
{
    sycl::queue q = TestUtils::get_test_queue();
    std::vector<int> src = {1, 2, 3};
    const dpl_experimental::device_array<int> d(src, q);

    EXPECT_TRUE(d.begin() == d.data(), "const begin != data");
    EXPECT_TRUE(d.end() == d.data() + d.size(), "const end != data + size");
    EXPECT_TRUE(d.size() == 3, "const size wrong");
    return true;
}

// =====================================================================
// Capacity tests
// =====================================================================

bool
test_resize_grow()
{
    sycl::queue q = TestUtils::get_test_queue();
    dpl_experimental::device_array<int> d({1, 2, 3}, q);
    d.resize(6, q);
    EXPECT_TRUE(d.size() == 6, "resize grow: wrong size");

    std::vector<int> result = d.to_vector(q);
    EXPECT_TRUE(result[0] == 1 && result[1] == 2 && result[2] == 3, "resize grow: original elements changed");
    return true;
}

bool
test_resize_grow_with_value()
{
    sycl::queue q = TestUtils::get_test_queue();
    dpl_experimental::device_array<int> d({1, 2, 3}, q);
    d.resize(6, 99, q);
    EXPECT_TRUE(d.size() == 6, "resize grow with value: wrong size");

    std::vector<int> result = d.to_vector(q);
    EXPECT_TRUE(result[0] == 1 && result[1] == 2 && result[2] == 3,
                "resize grow with value: original elements changed");
    EXPECT_TRUE(result[3] == 99 && result[4] == 99 && result[5] == 99,
                "resize grow with value: new elements not filled");
    return true;
}

bool
test_resize_shrink()
{
    sycl::queue q = TestUtils::get_test_queue();
    dpl_experimental::device_array<int> d({1, 2, 3, 4, 5}, q);
    d.resize(2, q);
    EXPECT_TRUE(d.size() == 2, "resize shrink: wrong size");

    std::vector<int> result = d.to_vector(q);
    std::vector<int> expected = {1, 2};
    EXPECT_TRUE(result == expected, "resize shrink: data mismatch");
    return true;
}

bool
test_reserve()
{
    sycl::queue q = TestUtils::get_test_queue();
    dpl_experimental::device_array<int> d({1, 2, 3}, q);
    d.reserve(1000);
    EXPECT_TRUE(d.size() == 3, "reserve: size should not change");
    EXPECT_TRUE(d.capacity() >= 1000, "reserve: capacity too small");

    std::vector<int> result = d.to_vector(q);
    std::vector<int> expected = {1, 2, 3};
    EXPECT_TRUE(result == expected, "reserve: data changed");
    return true;
}

bool
test_clear()
{
    sycl::queue q = TestUtils::get_test_queue();
    dpl_experimental::device_array<int> d({1, 2, 3}, q);
    d.clear();
    EXPECT_TRUE(d.size() == 0, "clear: size should be 0");
    EXPECT_TRUE(d.empty(), "clear: should be empty");
    return true;
}

// =====================================================================
// device_span tests
// =====================================================================

bool
test_span_basic()
{
    sycl::queue q = TestUtils::get_test_queue();
    dpl_experimental::device_array<int> d({1, 2, 3, 4, 5}, q);

    auto s = d.span();
    EXPECT_TRUE(s.data() == d.data(), "span: data pointer mismatch");
    EXPECT_TRUE(s.size() == d.size(), "span: size mismatch");
    EXPECT_TRUE(!s.empty(), "span: should not be empty");

    // Const span from non-const span (via converting constructor)
    dpl_experimental::device_span<const int> cs = d.span();
    EXPECT_TRUE(cs.data() == d.data(), "const span: data pointer mismatch");
    EXPECT_TRUE(cs.size() == d.size(), "const span: size mismatch");

    return true;
}

bool
test_span_subspan()
{
    sycl::queue q = TestUtils::get_test_queue();
    dpl_experimental::device_array<int> d({10, 20, 30, 40, 50}, q);

    auto s = d.span();
    auto sub = s.subspan(1, 3);
    EXPECT_TRUE(sub.size() == 3, "subspan: wrong size");
    EXPECT_TRUE(sub.data() == d.data() + 1, "subspan: wrong data pointer");
    return true;
}

bool
test_span_in_kernel()
{
    sycl::queue q = TestUtils::get_test_queue();
    dpl_experimental::device_array<int> d(10, 1, q);

    auto s = d.span();
    q.parallel_for<KernelSpanInKernel>(sycl::range<1>(s.size()), [=](sycl::id<1> i) {
         s[i] += static_cast<int>(i[0]);
     }).wait();

    std::vector<int> result = d.to_vector(q);
    bool all_correct = true;
    for (std::size_t i = 0; i < result.size(); ++i)
        all_correct &= (result[i] == static_cast<int>(1 + i));
    EXPECT_TRUE(all_correct, "span in kernel: wrong result");
    return true;
}

bool
test_span_const_in_kernel()
{
    sycl::queue q = TestUtils::get_test_queue();
    dpl_experimental::device_array<int> d_in({1, 2, 3, 4, 5}, q);
    dpl_experimental::device_array<int> d_out(5, q);

    dpl_experimental::device_span<const int> s_in = d_in.span();
    auto s_out = d_out.span();
    q.parallel_for<KernelSpanConstInKernel>(sycl::range<1>(s_in.size()), [=](sycl::id<1> i) {
         s_out[i] = s_in[i] * 2;
     }).wait();

    std::vector<int> result = d_out.to_vector(q);
    std::vector<int> expected = {2, 4, 6, 8, 10};
    EXPECT_TRUE(result == expected, "const span in kernel: wrong result");
    return true;
}

#    if _ENABLE_STD_RANGES_TESTING && TEST_CPP20_SPAN_PRESENT
bool
test_std_span_in_kernel()
{
    sycl::queue q = TestUtils::get_test_queue();
    dpl_experimental::device_array<int> d(10, 1, q);

    std::span<int> s(d.data(), d.size());
    q.parallel_for<KernelStdSpanInKernel>(sycl::range<1>(s.size()), [=](sycl::id<1> i) {
         s[i] += static_cast<int>(i);
     }).wait();

    std::vector<int> result = d.to_vector(q);
    bool all_correct = true;
    for (std::size_t i = 0; i < result.size(); ++i)
        all_correct &= (result[i] == static_cast<int>(1 + i));
    EXPECT_TRUE(all_correct, "std::span in kernel: wrong result");
    return true;
}

bool
test_std_span_const_in_kernel()
{
    sycl::queue q = TestUtils::get_test_queue();
    dpl_experimental::device_array<int> d_in({1, 2, 3, 4, 5}, q);
    dpl_experimental::device_array<int> d_out(5, q);

    std::span<const int> s_in(d_in.data(), d_in.size());
    std::span<int> s_out(d_out.data(), d_out.size());
    q.parallel_for<KernelStdSpanConstInKernel>(sycl::range<1>(s_in.size()), [=](sycl::id<1> i) {
         s_out[i] = s_in[i] * 2;
     }).wait();

    std::vector<int> result = d_out.to_vector(q);
    std::vector<int> expected = {2, 4, 6, 8, 10};
    EXPECT_TRUE(result == expected, "std::span const in kernel: wrong result");
    return true;
}
#    endif // _ENABLE_STD_RANGES_TESTING && TEST_CPP20_SPAN_PRESENT

#    if _ENABLE_RANGES_TESTING
bool
test_span_range_sort()
{
    sycl::queue q = TestUtils::get_test_queue();
    std::vector<int> src = {5, 3, 1, 4, 2};
    dpl_experimental::device_array<int> d(src, q);

    auto policy = oneapi::dpl::execution::make_device_policy<KernelSpanSort>(q);
    auto s = d.span();
    oneapi::dpl::experimental::ranges::sort(policy, s);

    std::vector<int> result = d.to_vector(q);
    std::vector<int> expected = {1, 2, 3, 4, 5};
    EXPECT_TRUE(result == expected, "span range sort: wrong result");
    return true;
}

bool
test_span_range_for_each()
{
    sycl::queue q = TestUtils::get_test_queue();
    dpl_experimental::device_array<int> d({1, 2, 3, 4, 5}, q);

    auto policy = oneapi::dpl::execution::make_device_policy<KernelSpanForEach>(q);
    auto s = d.span();
    oneapi::dpl::experimental::ranges::for_each(policy, s, [](int& x) { x *= 10; });

    std::vector<int> result = d.to_vector(q);
    std::vector<int> expected = {10, 20, 30, 40, 50};
    EXPECT_TRUE(result == expected, "span range for_each: wrong result");
    return true;
}

bool
test_span_range_transform()
{
    sycl::queue q = TestUtils::get_test_queue();
    dpl_experimental::device_array<int> d_in({1, 2, 3, 4, 5}, q);
    dpl_experimental::device_array<int> d_out(5, q);

    auto policy = oneapi::dpl::execution::make_device_policy<KernelSpanTransform>(q);
    auto s_in = d_in.span();
    auto s_out = d_out.span();
    oneapi::dpl::experimental::ranges::transform(policy, s_in, s_out, [](int x) { return x * 3; });

    std::vector<int> result = d_out.to_vector(q);
    std::vector<int> expected = {3, 6, 9, 12, 15};
    EXPECT_TRUE(result == expected, "span range transform: wrong result");
    return true;
}
#    endif // _ENABLE_RANGES_TESTING

// =====================================================================
// C++20 std ranges tests (oneapi::dpl::ranges)
// =====================================================================

#    if _ENABLE_STD_RANGES_TESTING

// --- Rvalue usage (span returned directly from .span()) ---

bool
test_span_std_ranges_sort()
{
    sycl::queue q = TestUtils::get_test_queue();
    std::vector<int> src = {5, 3, 1, 4, 2};
    dpl_experimental::device_array<int> d(src, q);

    auto policy = oneapi::dpl::execution::make_device_policy<KernelStdRangesSort>(q);
    oneapi::dpl::ranges::sort(policy, d.span());

    std::vector<int> result = d.to_vector(q);
    std::vector<int> expected = {1, 2, 3, 4, 5};
    EXPECT_TRUE(result == expected, "span std ranges sort (rvalue): wrong result");
    return true;
}

bool
test_span_std_ranges_for_each()
{
    sycl::queue q = TestUtils::get_test_queue();
    dpl_experimental::device_array<int> d({1, 2, 3, 4, 5}, q);

    auto policy = oneapi::dpl::execution::make_device_policy<KernelStdRangesForEach>(q);
    oneapi::dpl::ranges::for_each(policy, d.span(), [](int& x) { x *= 10; });

    std::vector<int> result = d.to_vector(q);
    std::vector<int> expected = {10, 20, 30, 40, 50};
    EXPECT_TRUE(result == expected, "span std ranges for_each (rvalue): wrong result");
    return true;
}

bool
test_span_std_ranges_transform()
{
    sycl::queue q = TestUtils::get_test_queue();
    dpl_experimental::device_array<int> d_in({1, 2, 3, 4, 5}, q);
    dpl_experimental::device_array<int> d_out(5, q);

    auto policy = oneapi::dpl::execution::make_device_policy<KernelStdRangesTransform>(q);
    oneapi::dpl::ranges::transform(policy, d_in.span(), d_out.span(), [](int x) { return x * 3; });

    std::vector<int> result = d_out.to_vector(q);
    std::vector<int> expected = {3, 6, 9, 12, 15};
    EXPECT_TRUE(result == expected, "span std ranges transform (rvalue): wrong result");
    return true;
}

// --- Lvalue usage (span stored in a local variable) ---
// With enable_view = true, views::all returns device_span directly (no ref_view).

bool
test_span_std_ranges_sort_lvalue()
{
    sycl::queue q = TestUtils::get_test_queue();
    std::vector<int> src = {5, 3, 1, 4, 2};
    dpl_experimental::device_array<int> d(src, q);

    auto policy = oneapi::dpl::execution::make_device_policy<KernelStdRangesSortLvalue>(q);
    auto s = d.span();
    oneapi::dpl::ranges::sort(policy, s);

    std::vector<int> result = d.to_vector(q);
    std::vector<int> expected = {1, 2, 3, 4, 5};
    EXPECT_TRUE(result == expected, "span std ranges sort (lvalue): wrong result");
    return true;
}

bool
test_span_std_ranges_for_each_lvalue()
{
    sycl::queue q = TestUtils::get_test_queue();
    dpl_experimental::device_array<int> d({1, 2, 3, 4, 5}, q);

    auto policy = oneapi::dpl::execution::make_device_policy<KernelStdRangesForEachLvalue>(q);
    auto s = d.span();
    oneapi::dpl::ranges::for_each(policy, s, [](int& x) { x *= 10; });

    std::vector<int> result = d.to_vector(q);
    std::vector<int> expected = {10, 20, 30, 40, 50};
    EXPECT_TRUE(result == expected, "span std ranges for_each (lvalue): wrong result");
    return true;
}

bool
test_span_std_ranges_transform_lvalue()
{
    sycl::queue q = TestUtils::get_test_queue();
    dpl_experimental::device_array<int> d_in({1, 2, 3, 4, 5}, q);
    dpl_experimental::device_array<int> d_out(5, q);

    auto policy = oneapi::dpl::execution::make_device_policy<KernelStdRangesTransformLvalue>(q);
    auto s_in = d_in.span();
    auto s_out = d_out.span();
    oneapi::dpl::ranges::transform(policy, s_in, s_out, [](int x) { return x * 3; });

    std::vector<int> result = d_out.to_vector(q);
    std::vector<int> expected = {3, 6, 9, 12, 15};
    EXPECT_TRUE(result == expected, "span std ranges transform (lvalue): wrong result");
    return true;
}

// --- Pipeline composition (span | view adaptor) ---

bool
test_span_std_ranges_sort_take()
{
    sycl::queue q = TestUtils::get_test_queue();
    std::vector<int> src = {5, 3, 1, 4, 2};
    dpl_experimental::device_array<int> d(src, q);

    auto policy = oneapi::dpl::execution::make_device_policy<KernelStdRangesSortTake>(q);
    oneapi::dpl::ranges::sort(policy, d.span() | std::views::take(3));

    std::vector<int> result = d.to_vector(q);
    // Only first 3 elements sorted: {5,3,1} -> {1,3,5}, rest unchanged
    std::vector<int> expected = {1, 3, 5, 4, 2};
    EXPECT_TRUE(result == expected, "span std ranges sort | take: wrong result");
    return true;
}

bool
test_span_std_ranges_for_each_reverse()
{
    sycl::queue q = TestUtils::get_test_queue();
    dpl_experimental::device_array<int> d({1, 2, 3, 4, 5}, q);

    auto policy = oneapi::dpl::execution::make_device_policy<KernelStdRangesForEachReverse>(q);
    oneapi::dpl::ranges::for_each(policy, d.span() | std::views::reverse, [](int& x) { x *= 10; });

    std::vector<int> result = d.to_vector(q);
    std::vector<int> expected = {10, 20, 30, 40, 50};
    EXPECT_TRUE(result == expected, "span std ranges for_each | reverse: wrong result");
    return true;
}

#        if !_PSTL_LIBSTDCXX_XPU_DROP_VIEW_BROKEN
bool
test_span_std_ranges_count_if_drop()
{
    sycl::queue q = TestUtils::get_test_queue();
    dpl_experimental::device_array<int> d({1, 2, 3, 4, 5}, q);

    auto policy = oneapi::dpl::execution::make_device_policy<KernelStdRangesCountIfDrop>(q);
    auto count = oneapi::dpl::ranges::count_if(policy, d.span() | std::views::drop(2), [](int x) { return x > 3; });

    EXPECT_TRUE(count == 2, "span std ranges count_if | drop: wrong result");
    return true;
}
#        endif // !_PSTL_LIBSTDCXX_XPU_DROP_VIEW_BROKEN

// --- std::span with raw USM device pointer (C++20 baseline comparison) ---

#        if TEST_CPP20_SPAN_PRESENT
bool
test_std_span_with_usm_sort()
{
    sycl::queue q = TestUtils::get_test_queue();
    std::vector<int> src = {5, 3, 1, 4, 2};
    dpl_experimental::device_array<int> d(src, q);

    auto policy = oneapi::dpl::execution::make_device_policy<KernelStdSpanSort>(q);
    std::span<int> s(d.data(), d.size());
    oneapi::dpl::ranges::sort(policy, s);

    std::vector<int> result = d.to_vector(q);
    std::vector<int> expected = {1, 2, 3, 4, 5};
    EXPECT_TRUE(result == expected, "std::span with USM sort: wrong result");
    return true;
}

bool
test_std_span_with_usm_for_each()
{
    sycl::queue q = TestUtils::get_test_queue();
    dpl_experimental::device_array<int> d({1, 2, 3, 4, 5}, q);

    auto policy = oneapi::dpl::execution::make_device_policy<KernelStdSpanForEach>(q);
    std::span<int> s(d.data(), d.size());
    oneapi::dpl::ranges::for_each(policy, s, [](int& x) { x *= 10; });

    std::vector<int> result = d.to_vector(q);
    std::vector<int> expected = {10, 20, 30, 40, 50};
    EXPECT_TRUE(result == expected, "std::span with USM for_each: wrong result");
    return true;
}
#        endif // TEST_CPP20_SPAN_PRESENT

#    endif // _ENABLE_STD_RANGES_TESTING

// =====================================================================
// Context / device access tests
// =====================================================================

bool
test_context_device_access()
{
    sycl::queue q = TestUtils::get_test_queue();
    dpl_experimental::device_array<int> d(10, q);
    EXPECT_TRUE(d.get_context() == q.get_context(), "context mismatch");
    EXPECT_TRUE(d.get_device() == q.get_device(), "device mismatch");
    return true;
}

// =====================================================================
// oneDPL algorithm integration tests
// =====================================================================

bool
test_transform()
{
    sycl::queue q = TestUtils::get_test_queue();
    std::vector<int> src(100);
    std::iota(src.begin(), src.end(), 0);
    dpl_experimental::device_array<int> input(src, q);
    dpl_experimental::device_array<int> output(100, q);

    auto policy = oneapi::dpl::execution::make_device_policy<KernelTransformOut>(q);
    std::transform(policy, input.begin(), input.end(), output.begin(), [](int x) { return x * 2; });

    std::vector<int> result = output.to_vector(q);
    bool all_correct = true;
    for (std::size_t i = 0; i < result.size(); ++i)
        all_correct &= (result[i] == static_cast<int>(i * 2));
    EXPECT_TRUE(all_correct, "transform: wrong result");
    return true;
}

bool
test_reduce()
{
    sycl::queue q = TestUtils::get_test_queue();
    std::vector<int> src(100, 1);
    dpl_experimental::device_array<int> d(src, q);

    auto policy = oneapi::dpl::execution::make_device_policy<KernelReduce>(q);
    int sum = std::reduce(policy, d.begin(), d.end(), 0);
    EXPECT_TRUE(sum == 100, "reduce: wrong result");
    return true;
}

bool
test_fill()
{
    sycl::queue q = TestUtils::get_test_queue();
    dpl_experimental::device_array<int> d(50, q);

    auto policy = oneapi::dpl::execution::make_device_policy<KernelFill>(q);
    std::fill(policy, d.begin(), d.end(), 7);

    std::vector<int> result = d.to_vector(q);
    bool all_correct = true;
    for (auto& v : result)
        all_correct &= (v == 7);
    EXPECT_TRUE(all_correct, "fill: wrong result");
    return true;
}

bool
test_copy_between_device_arrays()
{
    sycl::queue q = TestUtils::get_test_queue();
    std::vector<int> src = {1, 2, 3, 4, 5};
    dpl_experimental::device_array<int> d1(src, q);
    dpl_experimental::device_array<int> d2(5, q);

    auto policy = oneapi::dpl::execution::make_device_policy<KernelCopyDst>(q);
    std::copy(policy, d1.begin(), d1.end(), d2.begin());

    std::vector<int> result = d2.to_vector(q);
    EXPECT_TRUE(result == src, "copy between arrays: data mismatch");
    return true;
}

// =====================================================================
// Edge case tests
// =====================================================================

bool
test_zero_size()
{
    sycl::queue q = TestUtils::get_test_queue();
    dpl_experimental::device_array<int> d(static_cast<std::size_t>(0), q);
    EXPECT_TRUE(d.size() == 0, "zero size: wrong size");
    EXPECT_TRUE(d.empty(), "zero size: should be empty");
    EXPECT_TRUE(d.begin() == d.end(), "zero size: begin should equal end");

    std::vector<int> result = d.to_vector(q);
    EXPECT_TRUE(result.empty(), "zero size: to_vector should be empty");
    return true;
}

bool
test_large_allocation()
{
    sycl::queue q = TestUtils::get_test_queue();
    const std::size_t n = 1 << 20; // 1M elements
    dpl_experimental::device_array<int> d(n, 42, q);
    EXPECT_TRUE(d.size() == n, "large alloc: wrong size");
    int val = d.read(n - 1, q);
    EXPECT_TRUE(val == 42, "large alloc: last element wrong");
    return true;
}

// =====================================================================
// Struct type tests
// =====================================================================

bool
test_struct_construct_and_transfer()
{
    sycl::queue q = TestUtils::get_test_queue();
    std::vector<point_t> src = {{1.0f, 2.0f, 3.0f, 1}, {4.0f, 5.0f, 6.0f, 2}, {7.0f, 8.0f, 9.0f, 3}};
    dpl_experimental::device_array<point_t> d(src, q);
    std::vector<point_t> result = d.to_vector(q);
    EXPECT_TRUE(result == src, "struct construction + to_vector: data mismatch");
    return true;
}

bool
test_struct_read_write()
{
    sycl::queue q = TestUtils::get_test_queue();
    std::vector<point_t> src = {{1.0f, 2.0f, 3.0f, 10}, {4.0f, 5.0f, 6.0f, 20}};
    dpl_experimental::device_array<point_t> d(src, q);

    point_t p = d.read(0, q);
    EXPECT_TRUE(p == src[0], "struct read: wrong value");

    point_t new_p = {99.0f, 88.0f, 77.0f, 42};
    d.write(1, new_p, q);
    point_t p2 = d.read(1, q);
    EXPECT_TRUE(p2 == new_p, "struct write then read: wrong value");
    return true;
}

bool
test_struct_copy_move()
{
    sycl::queue q = TestUtils::get_test_queue();
    std::vector<point_t> src = {{1.0f, 2.0f, 3.0f, 1}, {4.0f, 5.0f, 6.0f, 2}};
    dpl_experimental::device_array<point_t> d1(src, q);

    dpl_experimental::device_array<point_t> d2(d1);
    std::vector<point_t> r2 = d2.to_vector(q);
    EXPECT_TRUE(r2 == src, "struct copy: data mismatch");
    EXPECT_TRUE(d1.data() != d2.data(), "struct copy: should be separate allocations");

    dpl_experimental::device_array<point_t> d3(std::move(d1));
    std::vector<point_t> r3 = d3.to_vector(q);
    EXPECT_TRUE(r3 == src, "struct move: data mismatch");
    EXPECT_TRUE(d1.empty(), "struct move: source should be empty");
    return true;
}

bool
test_struct_in_kernel()
{
    sycl::queue q = TestUtils::get_test_queue();
    std::vector<point_t> src = {{1.0f, 2.0f, 3.0f, 1}, {4.0f, 5.0f, 6.0f, 2}, {7.0f, 8.0f, 9.0f, 3}};
    dpl_experimental::device_array<point_t> d(src, q);

    point_t* ptr = d.data();
    q.parallel_for<KernelStructScale>(sycl::range<1>(d.size()), [=](sycl::id<1> i) {
         ptr[i].x *= 2.0f;
         ptr[i].y *= 2.0f;
         ptr[i].z *= 2.0f;
     }).wait();

    std::vector<point_t> result = d.to_vector(q);
    for (std::size_t i = 0; i < src.size(); ++i)
    {
        EXPECT_TRUE(result[i].x == src[i].x * 2.0f, "struct kernel: x wrong");
        EXPECT_TRUE(result[i].y == src[i].y * 2.0f, "struct kernel: y wrong");
        EXPECT_TRUE(result[i].z == src[i].z * 2.0f, "struct kernel: z wrong");
        EXPECT_TRUE(result[i].id == src[i].id, "struct kernel: id should be unchanged");
    }
    return true;
}

bool
test_struct_resize()
{
    sycl::queue q = TestUtils::get_test_queue();
    point_t fill_val = {0.0f, 0.0f, 0.0f, -1};
    std::vector<point_t> src = {{1.0f, 2.0f, 3.0f, 1}};
    dpl_experimental::device_array<point_t> d(src, q);

    d.resize(3, fill_val, q);
    EXPECT_TRUE(d.size() == 3, "struct resize: wrong size");

    std::vector<point_t> result = d.to_vector(q);
    EXPECT_TRUE(result[0] == src[0], "struct resize: original element changed");
    EXPECT_TRUE(result[1] == fill_val, "struct resize: new element not filled");
    EXPECT_TRUE(result[2] == fill_val, "struct resize: new element not filled");
    return true;
}

// =====================================================================
// Reserve data integrity test
// =====================================================================

bool
test_reserve_preserves_data()
{
    sycl::queue q = TestUtils::get_test_queue();
    std::vector<int> src = {10, 20, 30, 40, 50};
    dpl_experimental::device_array<int> d(src, q);

    d.reserve(1000);
    EXPECT_TRUE(d.size() == 5, "reserve preserves data: wrong size");
    EXPECT_TRUE(d.capacity() >= 1000, "reserve preserves data: capacity too small");

    std::vector<int> result = d.to_vector(q);
    EXPECT_TRUE(result == src, "reserve preserves data: data changed after reserve");

    d.reserve(500);
    result = d.to_vector(q);
    EXPECT_TRUE(result == src, "reserve preserves data: data changed after smaller reserve");
    return true;
}

bool
test_reserve_with_queue()
{
    sycl::queue q = TestUtils::get_test_queue();
    std::vector<int> src = {1, 2, 3};
    dpl_experimental::device_array<int> d(src, q);

    d.reserve(500, q);
    EXPECT_TRUE(d.size() == 3, "reserve(q): size should not change");
    EXPECT_TRUE(d.capacity() >= 500, "reserve(q): capacity too small");

    std::vector<int> result = d.to_vector(q);
    EXPECT_TRUE(result == src, "reserve(q): data changed");
    return true;
}

// =====================================================================
// General iterator assign tests
// =====================================================================

bool
test_assign_from_deque_iterators()
{
    sycl::queue q = TestUtils::get_test_queue();
    dpl_experimental::device_array<int> d(3, 0, q);
    std::deque<int> src = {100, 200, 300, 400};
    d.assign(src.begin(), src.end(), q);
    std::vector<int> result = d.to_vector(q);
    std::vector<int> expected(src.begin(), src.end());
    EXPECT_TRUE(result == expected, "assign from deque iterators: data mismatch");
    EXPECT_TRUE(d.size() == 4, "assign from deque iterators: wrong size");
    return true;
}

bool
test_assign_from_list_iterators()
{
    sycl::queue q = TestUtils::get_test_queue();
    dpl_experimental::device_array<int> d(2, 0, q);
    std::list<int> src = {5, 10, 15};
    d.assign(src.begin(), src.end(), q);
    std::vector<int> result = d.to_vector(q);
    std::vector<int> expected(src.begin(), src.end());
    EXPECT_TRUE(result == expected, "assign from list iterators: data mismatch");
    return true;
}

bool
test_assign_from_deque_no_queue()
{
    sycl::queue q = TestUtils::get_test_queue();
    dpl_experimental::device_array<int> d(3, 0, q);
    std::deque<int> src = {7, 8, 9};
    d.assign(src.begin(), src.end());
    std::vector<int> result = d.to_vector(q);
    std::vector<int> expected(src.begin(), src.end());
    EXPECT_TRUE(result == expected, "assign from deque (no queue): data mismatch");
    return true;
}

#endif // TEST_DPCPP_BACKEND_PRESENT

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    // Construction
    test_construct_uninitialized_from_queue();
    test_construct_uninitialized_from_ctx_dev();
    test_construct_with_value();
    test_construct_from_iterators();
    test_construct_from_initializer_list();
    test_construct_from_vector();
    test_copy_constructor();
    test_move_constructor();
    test_copy_assignment();
    test_move_assignment();

    // Host-device transfer
    test_to_vector();
    test_assign_from_pointers();
    test_assign_from_vector();
    test_read_write();
    test_async_read_write();
    test_async_bulk_transfer();

    // Iteration
    test_iterators_with_algorithm();
    test_data_pointer_in_kernel();
    test_const_iterators();

    // Capacity
    test_resize_grow();
    test_resize_grow_with_value();
    test_resize_shrink();
    test_reserve();
    test_clear();

    // device_span
    test_span_basic();
    test_span_subspan();
    test_span_in_kernel();
    test_span_const_in_kernel();
#    if _ENABLE_STD_RANGES_TESTING && TEST_CPP20_SPAN_PRESENT
    test_std_span_in_kernel();
    test_std_span_const_in_kernel();
#    endif
#    if _ENABLE_RANGES_TESTING
    test_span_range_sort();
    test_span_range_for_each();
    test_span_range_transform();
#    endif // _ENABLE_RANGES_TESTING

#    if _ENABLE_STD_RANGES_TESTING
    // Productized C++20 ranges — rvalue
    test_span_std_ranges_sort();
    test_span_std_ranges_for_each();
    test_span_std_ranges_transform();

    // Productized C++20 ranges — lvalue
    test_span_std_ranges_sort_lvalue();
    test_span_std_ranges_for_each_lvalue();
    test_span_std_ranges_transform_lvalue();

    // Productized C++20 ranges — pipeline composition
    test_span_std_ranges_sort_take();
    test_span_std_ranges_for_each_reverse();
#        if !_PSTL_LIBSTDCXX_XPU_DROP_VIEW_BROKEN
    test_span_std_ranges_count_if_drop();
#        endif

#        if TEST_CPP20_SPAN_PRESENT
    // std::span with raw USM device pointer (baseline comparison)
    test_std_span_with_usm_sort();
    test_std_span_with_usm_for_each();
#        endif // TEST_CPP20_SPAN_PRESENT
#    endif     // _ENABLE_STD_RANGES_TESTING

    // Context / device
    test_context_device_access();

    // Algorithm integration
    test_transform();
    test_reduce();
    test_fill();
    test_copy_between_device_arrays();

    // Edge cases
    test_zero_size();
    test_large_allocation();

    // Struct types
    test_struct_construct_and_transfer();
    test_struct_read_write();
    test_struct_copy_move();
    test_struct_in_kernel();
    test_struct_resize();

    // Reserve data integrity
    test_reserve_preserves_data();
    test_reserve_with_queue();

    // General iterator assign
    test_assign_from_deque_iterators();
    test_assign_from_list_iterators();
    test_assign_from_deque_no_queue();
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done();
}
