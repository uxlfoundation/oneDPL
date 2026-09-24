// -*- C++ -*-
//===-- find_or_wide_scan.pass.cpp ----------------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// This file incorporates work covered by the following copyright and permission
// notice:
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

// The find_or backend scans several contiguous elements per work item, but only above a size threshold that
// every other test stays below, so nothing otherwise exercises that path. Force the threshold to zero and
// vary the match position, which puts the match at every element of an iteration and at every alignment
// against the work-group size.
//
// Not covered here: batches longer than one iteration, which need an input far larger than a test can afford.
// Both size thresholds have to be lowered: equal and mismatch below read two ranges, and are held to the
// second one, so leaving it alone would route them to the narrow scan and the test would pass vacuously.
#define _ONEDPL_FIND_OR_WIDE_SCAN_MIN_SIZE 0
#define _ONEDPL_FIND_OR_WIDE_SCAN_MULTI_ELEM_MIN_SIZE 0

#include "support/test_config.h"

#include _PSTL_TEST_HEADER(execution)
#include _PSTL_TEST_HEADER(algorithm)

#include "support/utils.h"

// FPGA declines the wide scan outright, so its threshold ignores the overrides above and there is
// nothing here to cover.
#if TEST_DPCPP_BACKEND_PRESENT && !_ONEDPL_FPGA_DEVICE
#    define TEST_FIND_OR_WIDE_SCAN 1
#else
#    define TEST_FIND_OR_WIDE_SCAN 0
#endif

#if TEST_FIND_OR_WIDE_SCAN
#    include <algorithm>
#    include <cstdint>
#    include <functional>
#    include <vector>

// A range stand-in for the routing checks below: __value_t needs only a value_type.
template <typename _T>
struct __rng_of
{
    using value_type = _T;
};

template <typename _Brick, typename _Tag, typename... _Ranges>
inline constexpr bool __scans_wide =
    oneapi::dpl::__par_backend_hetero::__find_or_wide_scan_profitable<_Brick, _Tag, _Ranges...>();

using __or_tag = oneapi::dpl::__par_backend_hetero::__parallel_or_tag;
using __fwd_tag = oneapi::dpl::__par_backend_hetero::__parallel_find_forward_tag<std::size_t>;
using __bwd_tag = oneapi::dpl::__par_backend_hetero::__parallel_find_backward_tag<std::size_t>;
using __rng = __rng_of<int>;
using __cmp = std::less<int>;

static_assert(oneapi::dpl::__par_backend_hetero::__find_or_wide_scan_min_size_for<__rng>() == 0);
static_assert(oneapi::dpl::__par_backend_hetero::__find_or_wide_scan_min_size_for<__rng, __rng>() == 0);

// The wide scan is opted into per brick, so pin the routing of every brick that reaches __parallel_find_or.
// any_of / all_of / none_of, find / find_if / find_if_not, equal, mismatch, is_sorted, is_sorted_until:
static_assert(__scans_wide<oneapi::dpl::unseq_backend::single_match_pred<__cmp>, __or_tag, __rng>);
static_assert(__scans_wide<oneapi::dpl::unseq_backend::single_match_pred<__cmp>, __fwd_tag, __rng, __rng>);
// is_heap, is_heap_until: the predicate gathers a parent index.
static_assert(!__scans_wide<oneapi::dpl::unseq_backend::single_match_pred_by_idx<__cmp>, __or_tag, __rng>);
// find_end, search: the predicate loops over the needle.
static_assert(!__scans_wide<oneapi::dpl::unseq_backend::multiple_match_pred<__cmp>, __bwd_tag, __rng, __rng>);
// find_first_of:
static_assert(!__scans_wide<oneapi::dpl::unseq_backend::first_match_pred<__cmp>, __fwd_tag, __rng, __rng>);
// search_n:
static_assert(!__scans_wide<oneapi::dpl::unseq_backend::n_elem_match_pred<__cmp, int, std::size_t>, __fwd_tag, __rng>);
// includes:
static_assert(!__scans_wide<oneapi::dpl::unseq_backend::__brick_includes<std::size_t, std::size_t, __cmp,
                                                                         oneapi::dpl::identity, oneapi::dpl::identity>,
                            __or_tag, __rng, __rng>);

// The wide scan is also gated on element width and on how many elements the predicate reads per index, so pin
// both ends of the width window against both element counts. The widest element of any scanned range decides.
using __rng16 = __rng_of<std::uint16_t>;
using __rng64 = __rng_of<std::uint64_t>;
using __rng_zip_2_2 = __rng_of<oneapi::dpl::__internal::tuple<std::uint16_t, std::uint16_t>>;
using __rng_zip_4_8 = __rng_of<oneapi::dpl::__internal::tuple<std::uint32_t, std::uint64_t>>;
struct __elem16
{
    std::uint64_t __a, __b;
};
// 8 bytes is above the window, at every element count and every tag.
static_assert(!__scans_wide<oneapi::dpl::unseq_backend::single_match_pred<__cmp>, __or_tag, __rng64>);
static_assert(!__scans_wide<oneapi::dpl::unseq_backend::single_match_pred<__cmp>, __fwd_tag, __rng64, __rng64>);
static_assert(!__scans_wide<oneapi::dpl::unseq_backend::single_match_pred<__cmp>, __fwd_tag, __rng, __rng64>);
// A zip range reads one element per component, so it is held to the same width as two ranges.
static_assert(!__scans_wide<oneapi::dpl::unseq_backend::single_match_pred<__cmp>, __or_tag, __rng_zip_4_8>);
// Elements wider than the window are declined at every element count.
static_assert(!__scans_wide<oneapi::dpl::unseq_backend::single_match_pred<__cmp>, __or_tag, __rng_of<__elem16>>);
// 2 bytes is below the window, so only a presence check reading one element per index qualifies.
static_assert(__scans_wide<oneapi::dpl::unseq_backend::single_match_pred<__cmp>, __or_tag, __rng16>);
static_assert(!__scans_wide<oneapi::dpl::unseq_backend::single_match_pred<__cmp>, __fwd_tag, __rng16, __rng16>);
static_assert(!__scans_wide<oneapi::dpl::unseq_backend::single_match_pred<__cmp>, __or_tag, __rng_zip_2_2>);

// Every match position for a size a test can enumerate; above that an odd stride -- coprime with the scan
// width and the work-group size -- growing as the square of the size, so a device that needs a larger size to
// reach the wide scan does not multiply what this sweeps.
std::size_t
match_position_step(std::size_t __n)
{
    return __n <= 1024 ? 1 : std::max(std::size_t(37), (__n * __n) >> 23) | 1;
}

// One name per call below: these algorithms share the find_or kernels, so under explicit kernel names a
// single policy would give every call the same kernel name.
class __find_if_name;
class __find_end_name;
class __any_of_name;
class __none_of_name;
class __mismatch_name;

template <typename Policy>
void
test_at_size(Policy&& __exec, std::size_t __n)
{
    sycl::queue __q = __exec.queue();
    auto __exec_find_if = TestUtils::make_new_policy<__find_if_name>(__exec);
    auto __exec_find_end = TestUtils::make_new_policy<__find_end_name>(__exec);
    auto __exec_any_of = TestUtils::make_new_policy<__any_of_name>(__exec);
    auto __exec_none_of = TestUtils::make_new_policy<__none_of_name>(__exec);
    auto __exec_mismatch = TestUtils::make_new_policy<__mismatch_name>(__exec);
    std::vector<int> __host(__n, 0);
    int* __d = sycl::malloc_device<int>(__n, __q);
    auto __is_one = [](int __x) { return __x == 1; };

    const std::size_t __step = match_position_step(__n);

    for (std::size_t __pos = 0; __pos <= __n; __pos += __step)
    {
        std::fill(__host.begin(), __host.end(), 0);
        const bool __has_match = __pos < __n;
        if (__has_match)
            __host[__pos] = 1;
        // A decoy past the first match: a forward scan must ignore it, a backward scan must return it.
        const std::size_t __decoy = __has_match && __pos + 1 < __n ? __n - 1 : __pos;
        if (__has_match)
            __host[__decoy] = 1;
        __q.memcpy(__d, __host.data(), __n * sizeof(int)).wait();

        // Forward tag: the first match.
        EXPECT_TRUE(oneapi::dpl::find_if(__exec_find_if, __d, __d + __n, __is_one) == __d + __pos,
                    "wrong index from find_if");
        // Backward tag: find_end over a one-element needle returns the last match. Its brick declines the
        // wide scan, so this covers the narrow path.
        if (__n > 1)
        {
            const int __needle = 1;
            int* __nd = sycl::malloc_device<int>(1, __q);
            __q.memcpy(__nd, &__needle, sizeof(int)).wait();
            auto __expected = __has_match ? __d + __decoy : __d + __n;
            EXPECT_TRUE(oneapi::dpl::find_end(__exec_find_end, __d, __d + __n, __nd, __nd + 1) == __expected,
                        "wrong index from find_end");
            sycl::free(__nd, __q);
        }
        // Or tag: presence only.
        EXPECT_TRUE(oneapi::dpl::any_of(__exec_any_of, __d, __d + __n, __is_one) == __has_match,
                    "wrong result from any_of");
        EXPECT_TRUE(oneapi::dpl::none_of(__exec_none_of, __d, __d + __n, __is_one) == !__has_match,
                    "wrong result from none_of");
        // Two ranges over one allocation, so the scan reads two elements per index.
        EXPECT_TRUE(oneapi::dpl::mismatch(__exec_mismatch, __d, __d + __n, __d).first == __d + __n,
                    "wrong result from mismatch of a range with itself");
    }
    sycl::free(__d, __q);
}

// Two 8-byte ranges. The width gate routes them to the narrow scan, which no other test reaches at this
// width above the size threshold.
class __mismatch_8byte_name;
class __equal_8byte_name;

template <typename Policy>
void
test_two_8byte_ranges(Policy&& __exec, std::size_t __n)
{
    using _T = std::uint64_t;
    sycl::queue __q = __exec.queue();
    auto __exec_mismatch = TestUtils::make_new_policy<__mismatch_8byte_name>(__exec);
    auto __exec_equal = TestUtils::make_new_policy<__equal_8byte_name>(__exec);
    std::vector<_T> __host(__n, _T(1));
    _T* __d1 = sycl::malloc_device<_T>(__n, __q);
    _T* __d2 = sycl::malloc_device<_T>(__n, __q);
    __q.memcpy(__d1, __host.data(), __n * sizeof(_T)).wait();

    const std::size_t __step = match_position_step(__n);

    for (std::size_t __pos = 0; __pos <= __n; __pos += __step)
    {
        std::fill(__host.begin(), __host.end(), _T(1));
        const bool __differs = __pos < __n;
        if (__differs)
            __host[__pos] = _T(2);
        __q.memcpy(__d2, __host.data(), __n * sizeof(_T)).wait();

        EXPECT_TRUE(oneapi::dpl::mismatch(__exec_mismatch, __d1, __d1 + __n, __d2).first ==
                        (__differs ? __d1 + __pos : __d1 + __n),
                    "wrong index from mismatch of two 8-byte ranges");
        EXPECT_TRUE(oneapi::dpl::equal(__exec_equal, __d1, __d1 + __n, __d2) == !__differs,
                    "wrong result from equal of two 8-byte ranges");
    }
    sycl::free(__d1, __q);
    sycl::free(__d2, __q);
}

// A 2-byte element type; the cases above cover only 4 and 8 bytes. Only any_of and none_of take the wide
// scan at this width -- find_if, mismatch and equal are gated to the narrow one, and cover it here.
class __find_if_2byte_name;
class __any_of_2byte_name;
class __none_of_2byte_name;
class __mismatch_2byte_name;
class __equal_2byte_name;

template <typename Policy>
void
test_2byte_ranges(Policy&& __exec, std::size_t __n)
{
    using _T = std::uint16_t;
    sycl::queue __q = __exec.queue();
    auto __exec_find_if = TestUtils::make_new_policy<__find_if_2byte_name>(__exec);
    auto __exec_any_of = TestUtils::make_new_policy<__any_of_2byte_name>(__exec);
    auto __exec_none_of = TestUtils::make_new_policy<__none_of_2byte_name>(__exec);
    auto __exec_mismatch = TestUtils::make_new_policy<__mismatch_2byte_name>(__exec);
    auto __exec_equal = TestUtils::make_new_policy<__equal_2byte_name>(__exec);
    auto __is_one = [](_T __x) { return __x == _T(1); };
    std::vector<_T> __host(__n, _T(0));
    _T* __d1 = sycl::malloc_device<_T>(__n, __q);
    _T* __d2 = sycl::malloc_device<_T>(__n, __q);
    // The second range stays all-zero, so one match in the first serves every tag below.
    __q.memcpy(__d2, __host.data(), __n * sizeof(_T)).wait();

    const std::size_t __step = match_position_step(__n);

    for (std::size_t __pos = 0; __pos <= __n; __pos += __step)
    {
        std::fill(__host.begin(), __host.end(), _T(0));
        const bool __has_match = __pos < __n;
        if (__has_match)
            __host[__pos] = _T(1);
        __q.memcpy(__d1, __host.data(), __n * sizeof(_T)).wait();

        // Forward tag: the first match.
        EXPECT_TRUE(oneapi::dpl::find_if(__exec_find_if, __d1, __d1 + __n, __is_one) == __d1 + __pos,
                    "wrong index from find_if over 2-byte elements");
        // Or tag: presence only.
        EXPECT_TRUE(oneapi::dpl::any_of(__exec_any_of, __d1, __d1 + __n, __is_one) == __has_match,
                    "wrong result from any_of over 2-byte elements");
        EXPECT_TRUE(oneapi::dpl::none_of(__exec_none_of, __d1, __d1 + __n, __is_one) == !__has_match,
                    "wrong result from none_of over 2-byte elements");
        // Two ranges, so one iteration loads from two streams at 2 bytes each.
        EXPECT_TRUE(oneapi::dpl::mismatch(__exec_mismatch, __d1, __d1 + __n, __d2).first ==
                        (__has_match ? __d1 + __pos : __d1 + __n),
                    "wrong index from mismatch of two 2-byte ranges");
        EXPECT_TRUE(oneapi::dpl::equal(__exec_equal, __d1, __d1 + __n, __d2) == !__has_match,
                    "wrong result from equal of two 2-byte ranges");
    }
    sycl::free(__d1, __q);
    sycl::free(__d2, __q);
}
#endif // TEST_FIND_OR_WIDE_SCAN

int
main()
{
#if TEST_FIND_OR_WIDE_SCAN
    auto __policy = TestUtils::get_dpcpp_test_policy();
    // The wide scan lives on the multiple work-group path, which a size reaches only past the single work-group
    // path's reach, and that reach scales with the device's maximum work-group size.
    const std::size_t __beyond_one_wg =
        2 * oneapi::dpl::__internal::__max_work_group_size(__policy.queue(), std::size_t(4096)) *
        oneapi::dpl::__par_backend_hetero::__find_or_one_wg_max_elems_per_item;
    // Sizes that are and are not a multiple of the scan width. Only the largest takes the wide scan; the
    // rest cover the narrow one.
    for (std::size_t __n : {std::size_t(1), std::size_t(3), std::size_t(4), std::size_t(31), std::size_t(1024),
                            std::size_t(4095), __beyond_one_wg})
    {
        test_at_size(__policy, __n);
        test_two_8byte_ranges(__policy, __n);
        test_2byte_ranges(__policy, __n);
    }
#endif
    return TestUtils::done(TEST_FIND_OR_WIDE_SCAN);
}
