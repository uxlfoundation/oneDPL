// -*- C++ -*-
//===-- device_array.pass.cpp ---------------------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Container semantics of oneapi::dpl::experimental::device_array: construction,
// move/swap, the explicit transfer operations and their truncation rules.

#include "support/test_config.h"

#if TEST_DPCPP_BACKEND_PRESENT
#include <oneapi/dpl/experimental/device_array>
#endif

#include "support/utils.h"

// device_array is only available where oneapi::dpl::span is, which under C++17 requires the SYCL
// implementation to provide sycl::span.
#if TEST_DPCPP_BACKEND_PRESENT && TEST_SPAN_PRESENT
#    define TEST_DEVICE_ARRAY_PRESENT 1
#else
#    define TEST_DEVICE_ARRAY_PRESENT 0
#endif

#if TEST_DEVICE_ARRAY_PRESENT
#include "support/utils_sycl.h"
#include "support/utils_device_copyable.h"

#include <cstddef>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

namespace
{

template <typename _Tp>
using device_array = oneapi::dpl::experimental::device_array<_Tp>;

// Kernel names for the raw parallel_for submissions below, so that the test does not depend on
// unnamed lambda support.
template <typename _Tp, int _Id>
class writer_kernel;

// Every element type used here is constructible from int.
template <typename _Tp>
_Tp
make_value(int __i)
{
    return _Tp(__i);
}

// A host copy of the whole container that, unlike to_vector(), does not require _Tp to be default
// constructible, so the common tests can also run on a non-default-constructible element type.
template <typename _Tp>
std::vector<_Tp>
to_host(const device_array<_Tp>& __d, sycl::queue __q)
{
    std::vector<_Tp> __host_out(__d.size(), make_value<_Tp>(-1));
    __d.copy_to(oneapi::dpl::span<_Tp>{__host_out.data(), __host_out.size()}, __q);
    return __host_out;
}

template <typename _Fp>
bool
throws_out_of_range(_Fp __f)
{
    try
    {
        __f();
    }
    catch (const std::out_of_range&)
    {
        return true;
    }
    return false;
}

template <typename _Fp>
bool
throws_memory_allocation(_Fp __f)
{
    try
    {
        __f();
    }
    catch (const sycl::exception& __e)
    {
        return __e.code() == sycl::make_error_code(sycl::errc::memory_allocation);
    }
    return false;
}

template <typename _Tp>
std::vector<_Tp>
iota_host(std::size_t __n, int __start = 0)
{
    // Sizing the vector up front and assigning into it would require _Tp to be default constructible,
    // which the non-default-constructible element type these tests also run on is not. reserve() plus
    // push_back() performs a single allocation and constructs each element exactly once.
    std::vector<_Tp> __host_out;
    __host_out.reserve(__n);
    for (std::size_t __i = 0; __i < __n; ++__i)
        __host_out.push_back(make_value<_Tp>(__start + int(__i)));
    return __host_out;
}

template <typename _Tp>
void
test_type_traits()
{
    static_assert(!std::is_copy_constructible_v<device_array<_Tp>>, "device_array must not be copy constructible");
    static_assert(!std::is_copy_assignable_v<device_array<_Tp>>, "device_array must not be copy assignable");
    static_assert(std::is_move_constructible_v<device_array<_Tp>>, "device_array must be move constructible");
    static_assert(std::is_move_assignable_v<device_array<_Tp>>, "device_array must be move assignable");
    static_assert(!std::is_default_constructible_v<device_array<_Tp>>,
                  "device_array must not be default constructible");
    static_assert(std::is_same_v<typename device_array<_Tp>::value_type, _Tp>, "unexpected value_type");
    static_assert(std::is_same_v<typename device_array<_Tp>::size_type, std::size_t>, "unexpected size_type");

    // span() and oneapi::dpl::begin/end preserve constness and yield raw pointers.
    static_assert(std::is_same_v<decltype(std::declval<device_array<_Tp>&>().span()), oneapi::dpl::span<_Tp>>,
                  "span() on a device_array must return a span of _Tp");
    static_assert(
        std::is_same_v<decltype(std::declval<const device_array<_Tp>&>().span()), oneapi::dpl::span<const _Tp>>,
        "span() on a const device_array must return a span of const _Tp");
    static_assert(std::is_same_v<decltype(oneapi::dpl::begin(std::declval<device_array<_Tp>&>())), _Tp*>,
                  "oneapi::dpl::begin on a device_array must return a raw pointer");
    static_assert(std::is_same_v<decltype(oneapi::dpl::end(std::declval<device_array<_Tp>&>())), _Tp*>,
                  "oneapi::dpl::end on a device_array must return a raw pointer");
    static_assert(std::is_same_v<decltype(oneapi::dpl::begin(std::declval<const device_array<_Tp>&>())), const _Tp*>,
                  "oneapi::dpl::begin on a const device_array must return a raw pointer to const");
    static_assert(std::is_same_v<decltype(oneapi::dpl::end(std::declval<const device_array<_Tp>&>())), const _Tp*>,
                  "oneapi::dpl::end on a const device_array must return a raw pointer to const");

    // Device copyable, so a span may be captured by value in a kernel.
    static_assert(sycl::is_device_copyable_v<oneapi::dpl::span<_Tp>>, "oneapi::dpl::span must be device copyable");
    static_assert(sycl::is_device_copyable_v<oneapi::dpl::span<const _Tp>>,
                  "oneapi::dpl::span of const must be device copyable");

    static_assert(oneapi::dpl::is_indirectly_device_accessible_v<decltype(oneapi::dpl::begin(
                      std::declval<device_array<_Tp>&>()))>,
                  "oneapi::dpl::begin on a device_array must yield a device accessible iterator");
    static_assert(oneapi::dpl::is_indirectly_device_accessible_v<decltype(oneapi::dpl::begin(
                      std::declval<const device_array<_Tp>&>()))>,
                  "oneapi::dpl::begin on a const device_array must yield a device accessible iterator");
}

void
test_device_allocator(sycl::queue __q)
{
    using alloc_t = oneapi::dpl::experimental::device_allocator<int>;

    static_assert(std::is_same_v<alloc_t::value_type, int>, "device_allocator::value_type");
    static_assert(!std::is_default_constructible_v<alloc_t>, "device_allocator must not be default constructible");
    static_assert(std::is_copy_constructible_v<alloc_t>, "device_allocator must be copy constructible");
    static_assert(std::is_copy_assignable_v<alloc_t>, "device_allocator must be copy assignable");

    // Both constructor forms, as on usm_allocator.
    alloc_t __a(__q);
    alloc_t __b(__q.get_context(), __q.get_device());

    // The converting constructor carries the allocation target over.
    oneapi::dpl::experimental::device_allocator<double> __converted(__a);

    // Equality spans the value type and ignores the property list, but distinguishes the alignment, as
    // SYCL 2020 section 4.8.3.1 requires of usm_allocator.
    oneapi::dpl::experimental::device_allocator<int, 256> __aligned_a(__q);
    oneapi::dpl::experimental::device_allocator<double, 256> __aligned_b(__q);
    EXPECT_TRUE(__a == __b, "allocators on the same context and device must compare equal");
    EXPECT_TRUE(!(__a != __b), "operator!= must be the negation of operator==");
    EXPECT_TRUE(__a == __converted, "equality must span the value type");
    EXPECT_TRUE(__a != __aligned_a, "allocators with different alignments must compare unequal");
    EXPECT_TRUE(__aligned_a == __aligned_b, "equality must span the value type for a nonzero alignment");

    // A distinct context over the same device makes the allocators non interchangeable.
    sycl::context __other_ctx(__q.get_device());
    alloc_t __other(__other_ctx, __q.get_device());
    EXPECT_TRUE(__a != __other, "allocators on different contexts must compare unequal");
    EXPECT_TRUE(!(__a == __other), "operator== must be the negation of operator!=");

    // allocate/deallocate round trip, and the count == 0 short circuit.
    int* __p = __a.allocate(128);
    EXPECT_TRUE(__p != nullptr, "device_allocator::allocate returned null for a nonzero count");
    __a.deallocate(__p, 128);
    EXPECT_TRUE(__a.allocate(0) == nullptr, "device_allocator::allocate(0) must return nullptr");
    __a.deallocate(nullptr, 0); // must be a no-op

    // The aligned form dispatches to sycl::aligned_alloc_device.
    oneapi::dpl::experimental::device_allocator<int, 256> __aligned(__q);
    int* __ap = __aligned.allocate(64);
    EXPECT_TRUE(__ap != nullptr, "the aligned device_allocator returned null");
    EXPECT_TRUE(reinterpret_cast<std::uintptr_t>(__ap) % 256 == 0, "the aligned device_allocator ignored _Alignment");
    __aligned.deallocate(__ap, 64);

    EXPECT_TRUE(throws_memory_allocation([&] { __a.allocate(std::size_t(1) << 50); }),
                "device_allocator::allocate must throw errc::memory_allocation when allocation fails");

    // An unsupported (non power of two) alignment.
    oneapi::dpl::experimental::device_allocator<int, 3> __bad_alignment(__q);
    EXPECT_TRUE(throws_memory_allocation([&] { __bad_alignment.allocate(16); }),
                "the aligned device_allocator must throw when _Alignment is unsupported");

    EXPECT_TRUE(throws_memory_allocation([&] { device_array<int> __d(std::size_t(1) << 50, __q); }),
                "device_array construction must throw when its allocation fails");
}

// Uninitialized construction. The contents are deliberately never read
template <typename _Tp>
void
test_uninitialized_ctor(sycl::queue __q)
{
    const std::size_t __n = 64;

    device_array<_Tp> __d(__n, __q);
    EXPECT_EQ(__n, __d.size(), "uninitialized (count, queue): wrong size");
    EXPECT_TRUE(!__d.empty(), "uninitialized (count, queue): unexpectedly empty");
    EXPECT_EQ(__n, __d.span().size(), "uninitialized (count, queue): wrong span size");
    EXPECT_TRUE(__d.get_context() == __q.get_context(), "uninitialized (count, queue): wrong context");
    EXPECT_TRUE(__d.get_device() == __q.get_device(), "uninitialized (count, queue): wrong device");
    EXPECT_TRUE(oneapi::dpl::end(__d) - oneapi::dpl::begin(__d) == std::ptrdiff_t(__n),
                "uninitialized (count, queue): begin/end do not span the allocation");

    device_array<_Tp> __d2(__n, __q.get_context(), __q.get_device());
    EXPECT_EQ(__n, __d2.size(), "uninitialized (count, context, device): wrong size");
    EXPECT_TRUE(!__d2.empty(), "uninitialized (count, context, device): unexpectedly empty");
    EXPECT_EQ(__n, __d2.span().size(), "uninitialized (count, context, device): wrong span size");
    EXPECT_TRUE(__d2.get_context() == __q.get_context(), "uninitialized (count, context, device): wrong context");
    EXPECT_TRUE(__d2.get_device() == __q.get_device(), "uninitialized (count, context, device): wrong device");
}

// Fill construction
template <typename _Tp>
void
test_fill_ctor(sycl::queue __q)
{
    const std::size_t __n = 37;
    const _Tp __value = make_value<_Tp>(7);

    device_array<_Tp> __d(__n, __value, __q);
    const std::vector<_Tp> __expected(__n, __value);
    EXPECT_EQ_RANGES(__expected, to_host(__d, __q), "fill (count, value, queue): wrong contents");

    // The context+device form is verified through copy_to on an explicitly constructed queue.
    device_array<_Tp> __d2(__n, __value, __q.get_context(), __q.get_device());
    sycl::queue __explicit_q{__q.get_context(), __q.get_device()};
    std::vector<_Tp> __got(__n, make_value<_Tp>(-1));
    __d2.copy_to(oneapi::dpl::span<_Tp>{__got.data(), __got.size()}, __explicit_q);
    EXPECT_EQ_RANGES(__expected, __got, "fill (count, value, context, device): wrong contents");
}

// Construction from a host range: the implicit std::vector<_Tp>& -> span<const _Tp> conversion must
// work without naming the span at the call site.
template <typename _Tp>
void
test_host_range_ctor(sycl::queue __q)
{
    const std::size_t __n = 128;
    const std::vector<_Tp> __host = iota_host<_Tp>(__n);

    device_array<_Tp> __d(__host, __q);
    EXPECT_EQ(__n, __d.size(), "(host_vector, queue): wrong size");
    EXPECT_EQ_RANGES(__host, to_host(__d, __q), "(host_vector, queue): wrong contents");

    device_array<_Tp> __d2(__host, __q.get_context(), __q.get_device());
    EXPECT_EQ(__n, __d2.size(), "(host_vector, context, device): wrong size");
    EXPECT_EQ_RANGES(__host, to_host(__d2, __q), "(host_vector, context, device): wrong contents");
}

// A zero-element container allocates nothing and every operation on it is a no-op.
template <typename _Tp>
void
test_empty(sycl::queue __q)
{
    auto __check = [](device_array<_Tp>& __d, const char* __what) {
        EXPECT_EQ(std::size_t(0), __d.size(), __what);
        EXPECT_TRUE(__d.empty(), __what);
        EXPECT_TRUE(__d.span().empty(), __what);
        EXPECT_TRUE(oneapi::dpl::begin(__d) == oneapi::dpl::end(__d), __what);
    };

    device_array<_Tp> __d0(0, __q);
    __check(__d0, "device_array(0, queue) is not empty");

    device_array<_Tp> __d1(oneapi::dpl::span<const _Tp>{}, __q);
    __check(__d1, "device_array(empty span, queue) is not empty");

    // The bulk transfers truncate to zero and must not throw, even with a non-empty host side.
    std::vector<_Tp> __host = iota_host<_Tp>(4);
    const oneapi::dpl::span<const _Tp> __src{__host.data(), __host.size()};
    const oneapi::dpl::span<_Tp> __dst{__host.data(), __host.size()};
    EXPECT_EQ(std::size_t(0), __d0.copy_from(__src, __q), "copy_from on an empty device_array copied something");
    EXPECT_EQ(std::size_t(0), __d0.copy_to(__dst, __q), "copy_to on an empty device_array copied something");
    EXPECT_EQ(std::size_t(0), __d0.copy_from(__src), "queue-less copy_from on an empty device_array copied something");
    EXPECT_EQ(std::size_t(0), __d0.copy_to(__dst), "queue-less copy_to on an empty device_array copied something");
    EXPECT_EQ_RANGES(iota_host<_Tp>(4), __host, "transfers on an empty device_array modified the host buffer");

    // The single-element operations have no in-range position on an empty container: even 0 throws.
    EXPECT_TRUE(throws_out_of_range([&] { __d0.write_at(0, make_value<_Tp>(1), __q); }),
                "write_at(value, queue) on an empty device_array must throw");
    EXPECT_TRUE(throws_out_of_range([&] { __d0.write_at(0, make_value<_Tp>(1)); }),
                "write_at(value) on an empty device_array must throw");
    EXPECT_TRUE(throws_out_of_range([&] { __d0.read_at(0, __q); }),
                "read_at(0, queue) on an empty device_array must throw");
    EXPECT_TRUE(throws_out_of_range([&] { __d0.read_at(0); }), "read_at(0) on an empty device_array must throw");

    if constexpr (std::is_default_constructible_v<_Tp>)
    {
        EXPECT_TRUE(__d0.to_vector().empty(), "to_vector() on an empty device_array is not empty");
        EXPECT_TRUE(__d0.to_vector(__q).empty(), "to_vector(queue) on an empty device_array is not empty");
    }
}

// Move construction steals the pointer; the source retains its context and device.
template <typename _Tp>
void
test_move_ctor(sycl::queue __q)
{
    const std::size_t __n = 16;
    const std::vector<_Tp> __host = iota_host<_Tp>(__n);

    device_array<_Tp> __src(__host, __q);
    const _Tp* __src_ptr = oneapi::dpl::begin(__src);

    device_array<_Tp> __dst(std::move(__src));
    EXPECT_TRUE(oneapi::dpl::begin(__dst) == __src_ptr, "move ctor did not steal the allocation");
    EXPECT_EQ(__n, __dst.size(), "move ctor: wrong size in the target");
    EXPECT_EQ_RANGES(__host, to_host(__dst, __q), "move ctor: wrong contents in the target");

    EXPECT_TRUE(__src.empty(), "move ctor: the source is not empty");
    EXPECT_EQ(std::size_t(0), __src.size(), "move ctor: the source size is not zero");
}

// Move assignment, with differently sized operands so the contents prove the steal.
template <typename _Tp>
void
test_move_assign(sycl::queue __q)
{
    const std::vector<_Tp> __host_src = iota_host<_Tp>(24, 100);
    const std::vector<_Tp> __host_dst = iota_host<_Tp>(8, 0);

    device_array<_Tp> __src(__host_src, __q);
    device_array<_Tp> __dst(__host_dst, __q);
    const _Tp* __src_ptr = oneapi::dpl::begin(__src);

    __dst = std::move(__src);
    EXPECT_TRUE(oneapi::dpl::begin(__dst) == __src_ptr, "move assign did not steal the allocation");
    EXPECT_EQ(__host_src.size(), __dst.size(), "move assign: wrong size in the target");
    EXPECT_EQ_RANGES(__host_src, to_host(__dst, __q), "move assign: wrong contents in the target");

    EXPECT_TRUE(__src.empty(), "move assign: the source is not empty");
    EXPECT_TRUE(oneapi::dpl::begin(__src) == nullptr, "move assign: the source still holds a pointer");

    // Move assigning into a moved-from object is legal.
    device_array<_Tp> __other(__host_dst, __q);
    __src = std::move(__other);
    EXPECT_EQ_RANGES(__host_dst, to_host(__src, __q), "move assign into a moved-from target: wrong contents");
}

// Routed through a function taking two references so that the compiler cannot diagnose the self-move.
template <typename _Tp>
void
self_move_assign(device_array<_Tp>& __a, device_array<_Tp>& __b)
{
    __a = std::move(__b);
}

template <typename _Tp>
void
test_self_move_assign(sycl::queue __q)
{
    const std::vector<_Tp> __host = iota_host<_Tp>(12, 3);
    device_array<_Tp> __d(__host, __q);
    const _Tp* __ptr = oneapi::dpl::begin(__d);

    self_move_assign(__d, __d);

    EXPECT_EQ(__host.size(), __d.size(), "self move assign changed the size");
    EXPECT_TRUE(oneapi::dpl::begin(__d) == __ptr, "self move assign changed the allocation");
    EXPECT_EQ_RANGES(__host, to_host(__d, __q), "self move assign changed the contents");
}

// Member and free swap.
template <typename _Tp>
void
test_swap(sycl::queue __q)
{
    const std::vector<_Tp> __host_a = iota_host<_Tp>(10, 0);
    const std::vector<_Tp> __host_b = iota_host<_Tp>(20, 50);

    device_array<_Tp> __a(__host_a, __q);
    device_array<_Tp> __b(__host_b, __q);

    const _Tp* __ptr_a = oneapi::dpl::begin(__a);
    const _Tp* __ptr_b = oneapi::dpl::begin(__b);

    __a.swap(__b);
    EXPECT_TRUE(oneapi::dpl::begin(__a) == __ptr_b, "member swap did not exchange the allocations");
    EXPECT_TRUE(oneapi::dpl::begin(__b) == __ptr_a, "member swap did not exchange the allocations");
    EXPECT_EQ_RANGES(__host_b, to_host(__a, __q), "member swap: wrong contents in a");
    EXPECT_EQ_RANGES(__host_a, to_host(__b, __q), "member swap: wrong contents in b");

    // Free swap: found by ADL, since device_array and swap share a namespace.
    swap(__a, __b);
    EXPECT_TRUE(oneapi::dpl::begin(__a) == __ptr_a, "free swap did not exchange the allocations back");
    EXPECT_EQ_RANGES(__host_a, to_host(__a, __q), "free swap: wrong contents in a");
    EXPECT_EQ_RANGES(__host_b, to_host(__b, __q), "free swap: wrong contents in b");
}

// copy_to: the returned count, its truncation rules, and the offset precondition.
template <typename _Tp>
void
test_copy_to(sycl::queue __q)
{
    const std::size_t __n = 200;
    const std::vector<_Tp> __host = iota_host<_Tp>(__n);
    device_array<_Tp> __d(__host, __q);

    // Exact size, through all three overload forms.
    {
        std::vector<_Tp> __host_out(__n, make_value<_Tp>(-1));
        const oneapi::dpl::span<_Tp> __dst{__host_out.data(), __host_out.size()};
        EXPECT_EQ(__n, __d.copy_to(__dst, __q), "copy_to(dst, queue): wrong count");
        EXPECT_EQ_RANGES(__host, __host_out, "copy_to with an exactly sized destination");

        std::fill(__host_out.begin(), __host_out.end(), make_value<_Tp>(-1));
        EXPECT_EQ(__n, __d.copy_to(__dst, 0, __q), "copy_to(dst, 0, queue): wrong count");
        EXPECT_EQ_RANGES(__host, __host_out, "copy_to(dst, 0, queue): wrong contents");

        std::fill(__host_out.begin(), __host_out.end(), make_value<_Tp>(-1));
        EXPECT_EQ(__n, __d.copy_to(__dst), "copy_to(dst): wrong count");
        EXPECT_EQ_RANGES(__host, __host_out, "copy_to(dst): wrong contents");
    }

    // Destination larger than the container: truncates to size(), the tail is untouched.
    {
        const _Tp __sentinel = make_value<_Tp>(-9);
        std::vector<_Tp> __host_out(__n + 32, __sentinel);
        EXPECT_EQ(__n, __d.copy_to(oneapi::dpl::span<_Tp>{__host_out.data(), __host_out.size()}, __q),
                  "copy_to into a larger destination: wrong count");
        EXPECT_EQ_N(__host.begin(), __host_out.begin(), __n, "copy_to into a larger destination: wrong prefix");
        for (std::size_t __i = __n; __i < __host_out.size(); ++__i)
            EXPECT_TRUE(__host_out[__i] == __sentinel, "copy_to into a larger destination overwrote the tail");
    }

    // Destination smaller than the container: truncates to dst.size().
    {
        const std::size_t __m = 50;
        std::vector<_Tp> __host_out(__m, make_value<_Tp>(-1));
        EXPECT_EQ(__m, __d.copy_to(oneapi::dpl::span<_Tp>{__host_out.data(), __host_out.size()}, __q),
                  "copy_to into a smaller destination: wrong count");
        EXPECT_EQ_N(__host.begin(), __host_out.begin(), __m, "copy_to into a smaller destination: wrong contents");
    }

    // With a source offset: n = min(dst.size(), size() - src_offset).
    {
        const std::size_t __offset = 100;
        std::vector<_Tp> __host_out(__n, make_value<_Tp>(-1));
        EXPECT_EQ(__n - __offset,
                  __d.copy_to(oneapi::dpl::span<_Tp>{__host_out.data(), __host_out.size()}, __offset, __q),
                  "copy_to with src_offset: wrong count");
        EXPECT_EQ_N(__host.begin() + __offset, __host_out.begin(), __n - __offset,
                    "copy_to with src_offset: wrong contents");
    }

    // src_offset == size() is an empty transfer; past that is a precondition violation.
    {
        const _Tp __sentinel = make_value<_Tp>(-9);
        std::vector<_Tp> __host_out(8, __sentinel);
        const oneapi::dpl::span<_Tp> __dst{__host_out.data(), __host_out.size()};
        EXPECT_EQ(std::size_t(0), __d.copy_to(__dst, __n, __q), "copy_to with src_offset == size(): wrong count");
        EXPECT_EQ(std::size_t(0), __d.copy_to(__dst, __n), "queue-less copy_to with src_offset == size(): wrong count");
        for (const _Tp& __v : __host_out)
            EXPECT_TRUE(__v == __sentinel, "copy_to with src_offset == size() copied something");

        // out of range offset must throw
        EXPECT_TRUE(throws_out_of_range([&] { __d.copy_to(__dst, __n + 1, __q); }),
                    "copy_to with src_offset > size() must throw");
        EXPECT_TRUE(throws_out_of_range([&] { __d.copy_to(__dst, 10 * __n, __q); }),
                    "copy_to with a far out-of-range src_offset must throw");
        EXPECT_TRUE(throws_out_of_range([&] { __d.copy_to(__dst, __n + 1); }),
                    "queue-less copy_to with src_offset > size() must throw");
        for (const _Tp& __v : __host_out)
            EXPECT_TRUE(__v == __sentinel, "a throwing copy_to copied something");
    }
}

// copy_from, mirroring the copy_to matrix.
template <typename _Tp>
void
test_copy_from(sycl::queue __q)
{
    const std::size_t __n = 200;
    const _Tp __background = make_value<_Tp>(-5);

    // Exact size, through all three overload forms.
    {
        device_array<_Tp> __d(__n, __background, __q);
        const std::vector<_Tp> __host = iota_host<_Tp>(__n);
        const oneapi::dpl::span<const _Tp> __src{__host.data(), __host.size()};

        EXPECT_EQ(__n, __d.copy_from(__src, __q), "copy_from(src, queue): wrong count");
        EXPECT_EQ_RANGES(__host, to_host(__d, __q), "copy_from with an exactly sized source");

        device_array<_Tp> __d2(__n, __background, __q);
        EXPECT_EQ(__n, __d2.copy_from(__src, 0, __q), "copy_from(src, 0, queue): wrong count");
        EXPECT_EQ_RANGES(__host, to_host(__d2, __q), "copy_from(src, 0, queue): wrong contents");

        device_array<_Tp> __d3(__n, __background, __q);
        EXPECT_EQ(__n, __d3.copy_from(__src), "copy_from(src): wrong count");
        EXPECT_EQ_RANGES(__host, to_host(__d3, __q), "copy_from(src): wrong contents");
    }

    // Source larger than the container: truncates to size().
    {
        device_array<_Tp> __d(__n, __background, __q);
        const std::vector<_Tp> __host = iota_host<_Tp>(__n + 32);
        EXPECT_EQ(__n, __d.copy_from(oneapi::dpl::span<const _Tp>{__host.data(), __host.size()}, __q),
                  "copy_from from a larger source: wrong count");
        const std::vector<_Tp> __got = to_host(__d, __q);
        EXPECT_EQ(__n, __got.size(), "copy_from from a larger source changed the container size");
        EXPECT_EQ_N(__host.begin(), __got.begin(), __n, "copy_from from a larger source: wrong contents");
    }

    // Source smaller than the container: only the prefix is written.
    {
        device_array<_Tp> __d(__n, __background, __q);
        const std::size_t __m = 50;
        const std::vector<_Tp> __host = iota_host<_Tp>(__m);
        EXPECT_EQ(__m, __d.copy_from(oneapi::dpl::span<const _Tp>{__host.data(), __host.size()}, __q),
                  "copy_from from a smaller source: wrong count");
        const std::vector<_Tp> __got = to_host(__d, __q);
        EXPECT_EQ_N(__host.begin(), __got.begin(), __m, "copy_from from a smaller source: wrong prefix");
        for (std::size_t __i = __m; __i < __n; ++__i)
            EXPECT_TRUE(__got[__i] == __background, "copy_from from a smaller source overwrote the tail");
    }

    // With a destination offset: n = min(src.size(), size() - dst_offset).
    {
        device_array<_Tp> __d(__n, __background, __q);
        const std::size_t __offset = 100;
        const std::vector<_Tp> __host = iota_host<_Tp>(__n, 1000);
        EXPECT_EQ(__n - __offset,
                  __d.copy_from(oneapi::dpl::span<const _Tp>{__host.data(), __host.size()}, __offset, __q),
                  "copy_from with dst_offset: wrong count");
        const std::vector<_Tp> __got = to_host(__d, __q);
        for (std::size_t __i = 0; __i < __offset; ++__i)
            EXPECT_TRUE(__got[__i] == __background, "copy_from with dst_offset wrote before the offset");
        EXPECT_EQ_N(__host.begin(), __got.begin() + __offset, __n - __offset,
                    "copy_from with dst_offset: wrong contents");
    }

    // dst_offset == size() is an empty transfer; past that is a precondition violation.
    {
        device_array<_Tp> __d(__n, __background, __q);
        const std::vector<_Tp> __host = iota_host<_Tp>(8, 7);
        const oneapi::dpl::span<const _Tp> __src{__host.data(), __host.size()};

        EXPECT_EQ(std::size_t(0), __d.copy_from(__src, __n, __q), "copy_from with dst_offset == size(): wrong count");
        EXPECT_EQ(std::size_t(0), __d.copy_from(__src, __n),
                  "queue-less copy_from with dst_offset == size(): wrong count");

        // offset out of range must throw
        EXPECT_TRUE(throws_out_of_range([&] { __d.copy_from(__src, __n + 1, __q); }),
                    "copy_from with dst_offset > size() must throw");
        EXPECT_TRUE(throws_out_of_range([&] { __d.copy_from(__src, 10 * __n, __q); }),
                    "copy_from with a far out-of-range dst_offset must throw");
        EXPECT_TRUE(throws_out_of_range([&] { __d.copy_from(__src, __n + 1); }),
                    "queue-less copy_from with dst_offset > size() must throw");

        const std::vector<_Tp> __got = to_host(__d, __q);
        for (const _Tp& __v : __got)
            EXPECT_TRUE(__v == __background, "copy_from with an out-of-range dst_offset wrote something");
    }
}

// read_at, both overloads. Also runs for a non-default-constructible element type, which is what the
// __lazy_ctor_storage in the implementation is for.
template <typename _Tp>
void
test_read_at(sycl::queue __q)
{
    const std::size_t __n = 100;
    const std::vector<_Tp> __host = iota_host<_Tp>(__n, 5);
    device_array<_Tp> __d(__host, __q);

    EXPECT_TRUE(__d.read_at(0, __q) == __host[0], "read_at(0, queue): wrong value");
    EXPECT_TRUE(__d.read_at(__n / 2, __q) == __host[__n / 2], "read_at(middle, queue): wrong value");
    EXPECT_TRUE(__d.read_at(__n - 1, __q) == __host[__n - 1], "read_at(last, queue): wrong value");

    EXPECT_TRUE(__d.read_at(0) == __host[0], "read_at(0): wrong value");
    EXPECT_TRUE(__d.read_at(__n / 2) == __host[__n / 2], "read_at(middle): wrong value");
    EXPECT_TRUE(__d.read_at(__n - 1) == __host[__n - 1], "read_at(last): wrong value");

    // A single element is addressed, so pos == size() is already out of range.
    EXPECT_TRUE(throws_out_of_range([&] { __d.read_at(__n, __q); }), "read_at(size(), queue) must throw");
    EXPECT_TRUE(throws_out_of_range([&] { __d.read_at(__n); }), "read_at(size()) must throw");
    EXPECT_TRUE(throws_out_of_range([&] { __d.read_at(10 * __n, __q); }),
                "read_at with a far out-of-range position must throw");
}

// Single-element copy_from, both overloads.
template <typename _Tp>
void
test_single_element_write(sycl::queue __q)
{
    const std::size_t __n = 32;
    const _Tp __background = make_value<_Tp>(0);

    device_array<_Tp> __d(__n, __background, __q);
    const _Tp __v1 = make_value<_Tp>(11);
    const _Tp __v2 = make_value<_Tp>(22);

    __d.write_at(5, __v1, __q);
    __d.write_at(6, __v2);
    // The queue overload with a defaulted offset writes element 0.
    const _Tp __v0 = make_value<_Tp>(33);
    __d.write_at(0, __v0, __q);

    const std::vector<_Tp> __got = to_host(__d, __q);
    EXPECT_TRUE(__got[0] == __v0, "write_at(offset, value, queue): wrong value written");
    EXPECT_TRUE(__got[5] == __v1, "write_at(offset, value, queue): wrong value written");
    EXPECT_TRUE(__got[6] == __v2, "write_at(offset, value): wrong value written");
    for (std::size_t __i = 0; __i < __n; ++__i)
    {
        if (__i != 0 && __i != 5 && __i != 6)
            EXPECT_TRUE(__got[__i] == __background, "write_at(offset, value) disturbed a neighbor");
    }

    // A single element is addressed, so offset == size() is already out of range.
    EXPECT_TRUE(throws_out_of_range([&] { __d.write_at(__n, make_value<_Tp>(99), __q); }),
                "write_at(size(), value, queue) must throw");
    EXPECT_TRUE(throws_out_of_range([&] { __d.write_at(__n + 7, make_value<_Tp>(99)); }),
                "write_at(offset, value) with an out-of-range offset must throw");
    EXPECT_EQ_RANGES(__got, to_host(__d, __q), "a throwing write_at(offset, value) wrote something");
}

// to_vector, both overloads. Requires a default-constructible element type.
template <typename _Tp>
void
test_to_vector(sycl::queue __q)
{
    const std::size_t __n = 64;
    const std::vector<_Tp> __host = iota_host<_Tp>(__n, 2);
    device_array<_Tp> __d(__host, __q);

    EXPECT_EQ_RANGES(__host, __d.to_vector(__q), "to_vector(queue): wrong contents");
    EXPECT_EQ_RANGES(__host, __d.to_vector(), "to_vector(): wrong contents");

    device_array<_Tp> __empty(0, __q);
    EXPECT_TRUE(__empty.to_vector().empty(), "to_vector() on an empty container is not empty");
}

// Device-to-device transfers, deep copy through the span constructor, and a subrange copy, all of
// which lean on the span<_Tp> -> span<const _Tp> conversion.
template <typename _Tp>
void
test_device_to_device(sycl::queue __q)
{
    const std::size_t __n = 80;
    const std::vector<_Tp> __host = iota_host<_Tp>(__n, 1);
    device_array<_Tp> __d(__host, __q);

    // copy_from taking another container's span.
    device_array<_Tp> __d2(__n, make_value<_Tp>(0), __q);
    EXPECT_EQ(__n, __d2.copy_from(__d.span(), __q), "device-to-device copy_from: wrong count");
    EXPECT_EQ_RANGES(__host, to_host(__d2, __q), "device-to-device copy_from: wrong contents");

    // Deep copy through the span constructor.
    device_array<_Tp> __copy(__d.span(), __q);
    EXPECT_TRUE(oneapi::dpl::begin(__copy) != oneapi::dpl::begin(__d), "the span ctor did not allocate new storage");
    EXPECT_EQ_RANGES(__host, to_host(__copy, __q), "the span ctor produced wrong contents");

    // Mutating the source must not be visible through the copy.
    __d.write_at(0, make_value<_Tp>(999), __q);
    EXPECT_EQ_RANGES(__host, to_host(__copy, __q), "the span ctor produced a shallow copy");

    // Subrange.
    const std::size_t __k = 25;
    device_array<_Tp> __head(__d.span().subspan(0, __k), __q);
    EXPECT_EQ(__k, __head.size(), "subspan ctor: wrong size");
    const std::vector<_Tp> __expected_head = to_host(__d, __q);
    EXPECT_EQ_N(__expected_head.begin(), to_host(__head, __q).begin(), __k, "subspan ctor: wrong contents");
}

// A container that never saw a queue, exercised through every queue-less overload.
template <typename _Tp>
void
test_queueless_path(sycl::queue __q)
{
    const std::size_t __n = 40;
    const std::vector<_Tp> __host = iota_host<_Tp>(__n, 4);

    device_array<_Tp> __d(__n, make_value<_Tp>(0), __q.get_context(), __q.get_device());
    EXPECT_TRUE(__d.get_context() == __q.get_context(), "queue-less container: wrong context");

    EXPECT_EQ(__n, __d.copy_from(oneapi::dpl::span<const _Tp>{__host.data(), __host.size()}),
              "queue-less copy_from: wrong count");

    std::vector<_Tp> __host_out(__n, make_value<_Tp>(-1));
    EXPECT_EQ(__n, __d.copy_to(oneapi::dpl::span<_Tp>{__host_out.data(), __host_out.size()}),
              "queue-less copy_to: wrong count");
    EXPECT_EQ_RANGES(__host, __host_out, "queue-less copy_from/copy_to round trip");

    EXPECT_TRUE(__d.read_at(3) == __host[3], "queue-less read_at: wrong value");

    __d.write_at(3, make_value<_Tp>(77));
    EXPECT_TRUE(__d.read_at(3) == make_value<_Tp>(77), "queue-less write_at(offset, value) did not take effect");

    if constexpr (std::is_default_constructible_v<_Tp>)
    {
        const std::vector<_Tp> __v = __d.to_vector();
        EXPECT_EQ(__n, __v.size(), "queue-less to_vector: wrong size");
        EXPECT_TRUE(__v[3] == make_value<_Tp>(77), "queue-less to_vector: wrong value");
    }
}

// depends_on against an out-of-order queue, where a transfer can only be ordered after a kernel
// through the event. The iteration count makes an accidental pass unlikely.
void
test_depends_on(sycl::queue __q)
{
    sycl::queue __ooo_q{__q.get_context(), __q.get_device()};
    EXPECT_TRUE(!__ooo_q.is_in_order(), "the test queue for depends_on must be out of order");

    const std::size_t __n = 4096;
    const int __iterations = 50;

    for (int __it = 0; __it < __iterations; ++__it)
    {
        device_array<int> __d(__n, -1, __ooo_q);

        // copy_to must observe the kernel's writes.
        sycl::event __e = __ooo_q.parallel_for<writer_kernel<int, 0>>(
            sycl::range<1>(__n), [__s = __d.span()](sycl::id<1> __i) { __s[__i] = int(__i.get(0)); });

        std::vector<int> __host_out(__n, -1);
        __d.copy_to(oneapi::dpl::span<int>{__host_out.data(), __host_out.size()}, __ooo_q, __e);
        for (std::size_t __i = 0; __i < __n; ++__i)
            EXPECT_EQ(int(__i), __host_out[__i], "copy_to did not wait for the event it depends on");

        // read_at must observe them as well.
        sycl::event __e2 = __ooo_q.parallel_for<writer_kernel<int, 1>>(
            sycl::range<1>(__n), [__s = __d.span()](sycl::id<1> __i) { __s[__i] = int(__i.get(0)) + 1000; });
        EXPECT_EQ(1000 + int(__n / 2), __d.read_at(__n / 2, __ooo_q, __e2),
                  "read_at did not wait for the event it depends on");

        // copy_from must not overwrite the allocation before the kernel has read it.
        sycl::event __e3 = __ooo_q.parallel_for<writer_kernel<int, 2>>(
            sycl::range<1>(__n), [__s = __d.span()](sycl::id<1> __i) { __s[__i] = __s[__i] * 2; });
        const std::vector<int> __host(__n, 42);
        __d.copy_from(oneapi::dpl::span<const int>{__host.data(), __host.size()}, 0, __ooo_q, __e3);
        EXPECT_EQ_RANGES(__host, __d.to_vector(__ooo_q), "copy_from did not take effect after its dependency");
    }
}

template <typename _Tp>
void
test_all_common(sycl::queue __q)
{
    test_type_traits<_Tp>();
    test_uninitialized_ctor<_Tp>(__q);
    test_fill_ctor<_Tp>(__q);
    test_host_range_ctor<_Tp>(__q);
    test_empty<_Tp>(__q);
    test_move_ctor<_Tp>(__q);
    test_move_assign<_Tp>(__q);
    test_self_move_assign<_Tp>(__q);
    test_swap<_Tp>(__q);
    test_copy_to<_Tp>(__q);
    test_copy_from<_Tp>(__q);
    test_read_at<_Tp>(__q);
    test_single_element_write<_Tp>(__q);
    test_device_to_device<_Tp>(__q);
    test_queueless_path<_Tp>(__q);
}

} // namespace
#endif // TEST_DEVICE_ARRAY_PRESENT

int
main()
{
#if TEST_DEVICE_ARRAY_PRESENT
    sycl::queue q = TestUtils::get_test_queue();

    test_device_allocator(q);

    test_all_common<int>(q);
    test_all_common<float>(q);
    // Only to_vector() requires default constructibility, so everything else must work without it.
    test_all_common<TestUtils::NoDefaultCtorWrapper<int>>(q);

    test_to_vector<int>(q);
    test_to_vector<float>(q);

    test_depends_on(q);
#endif // TEST_DEVICE_ARRAY_PRESENT

    return TestUtils::done(TEST_DEVICE_ARRAY_PRESENT);
}
