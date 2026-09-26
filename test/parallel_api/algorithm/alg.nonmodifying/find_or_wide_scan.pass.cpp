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
// vary the match position, which puts the match at every element of an iteration. Both thresholds have to be
// zeroed: equal and mismatch read two ranges, and are held to the second one.
#define _ONEDPL_FIND_OR_WIDE_SCAN_MIN_SIZE 0
#define _ONEDPL_FIND_OR_WIDE_SCAN_MULTI_ELEM_MIN_SIZE 0

#include "support/test_config.h"

#include _PSTL_TEST_HEADER(execution)
#include _PSTL_TEST_HEADER(algorithm)

#include "support/utils.h"

// FPGA declines the wide scan outright, so its threshold ignores the overrides above and there is
// nothing here to cover.
#if TEST_DPCPP_BACKEND_PRESENT && !_ONEDPL_FPGA_DEVICE && !_ONEDPL_FPGA_EMU
#    define TEST_FIND_OR_WIDE_SCAN 1
#else
#    define TEST_FIND_OR_WIDE_SCAN 0
#endif

#if TEST_FIND_OR_WIDE_SCAN
#    include "support/sycl_alloc_utils.h"

#    include <algorithm>
#    include <cstddef>
#    include <cstdint>
#    include <functional>
#    include <vector>

template <typename _Brick, typename _Tag, typename... _Ranges>
inline constexpr bool __scans_wide =
    oneapi::dpl::__par_backend_hetero::__find_or_wide_scan_profitable<_Brick, _Tag, _Ranges...>();

using __or_tag = oneapi::dpl::__par_backend_hetero::__parallel_or_tag;
using __fwd_tag = oneapi::dpl::__par_backend_hetero::__parallel_find_forward_tag<std::size_t>;
using __bwd_tag = oneapi::dpl::__par_backend_hetero::__parallel_find_backward_tag<std::size_t>;
// The routing checks below need each range's value type and how the range is read, so they use the types the
// entry paths build: a passed-directly iterator becomes guard_view.
template <typename _T>
using __rng_of = oneapi::dpl::__ranges::guard_view<_T*>;
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
using __rng8 = __rng_of<std::uint8_t>;
using __rng16 = __rng_of<std::uint16_t>;
using __rng64 = __rng_of<std::uint64_t>;
// A zip range holds views, not iterators.
template <typename... _Ts>
using __zip_of = oneapi::dpl::__ranges::zip_view<__rng_of<_Ts>...>;
using __rng_zip_2_2 = __zip_of<std::uint16_t, std::uint16_t>;
using __rng_zip_4_8 = __zip_of<std::uint32_t, std::uint64_t>;
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
// 1 byte is below the presence check's own floor, so nothing admits it.
static_assert(!__scans_wide<oneapi::dpl::unseq_backend::single_match_pred<__cmp>, __or_tag, __rng8>);

// The width above is the value type's, which a view can report without loading it: both ranges below say 4
// bytes while one loads 8 and the other gathers, so both are declined whatever the width says.
struct __narrowing
{
    int
    operator()(std::uint64_t __x) const
    {
        return int(__x);
    }
};
struct __pair_swap
{
    std::size_t
    operator()(std::size_t __i) const
    {
        return __i ^ 1;
    }
};
using __rng_transform = oneapi::dpl::__ranges::transform_view_simple<__rng64, __narrowing>;
using __rng_permutation = oneapi::dpl::__ranges::permutation_view_simple<__rng, __pair_swap>;
static_assert(sizeof(oneapi::dpl::__internal::__value_t<__rng_transform>) == 4);
static_assert(sizeof(oneapi::dpl::__internal::__value_t<__rng_permutation>) == 4);
static_assert(!__scans_wide<oneapi::dpl::unseq_backend::single_match_pred<__cmp>, __or_tag, __rng_transform>);
static_assert(!__scans_wide<oneapi::dpl::unseq_backend::single_match_pred<__cmp>, __or_tag, __rng_permutation>);
// take and drop only move the ends, and is_sorted reaches the wide scan through them, so they stay admitted.
using __rng_take = oneapi::dpl::__ranges::take_view_simple<__rng, std::ptrdiff_t>;
using __rng_drop = oneapi::dpl::__ranges::drop_view_simple<__rng, std::ptrdiff_t>;
static_assert(__scans_wide<oneapi::dpl::unseq_backend::single_match_pred<__cmp>, __fwd_tag, __rng_take, __rng_drop>);

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
class __mismatch_self_name;
class __mismatch_name;
class __equal_name;

template <typename Policy>
void
test_at_size(Policy&& __exec, std::size_t __n)
{
    sycl::queue __q = __exec.queue();
    auto __exec_find_if = TestUtils::make_new_policy<__find_if_name>(__exec);
    auto __exec_find_end = TestUtils::make_new_policy<__find_end_name>(__exec);
    auto __exec_any_of = TestUtils::make_new_policy<__any_of_name>(__exec);
    auto __exec_none_of = TestUtils::make_new_policy<__none_of_name>(__exec);
    auto __exec_mismatch_self = TestUtils::make_new_policy<__mismatch_self_name>(__exec);
    auto __exec_mismatch = TestUtils::make_new_policy<__mismatch_name>(__exec);
    auto __exec_equal = TestUtils::make_new_policy<__equal_name>(__exec);
    std::vector<int> __host(__n, 0);
    std::vector<int> __host2(__n, 0);
    TestUtils::usm_data_transfer<sycl::usm::alloc::device, int> __dt(__q, __n);
    TestUtils::usm_data_transfer<sycl::usm::alloc::device, int> __dt2(__q, __n);
    int __needle = 1;
    TestUtils::usm_data_transfer<sycl::usm::alloc::device, int> __needle_dt(__q, &__needle, std::size_t(1));
    int* __d = __dt.get_data();
    int* __d2 = __dt2.get_data();
    int* __nd = __needle_dt.get_data();
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
        // The second range differs from the first at __pos alone, so the two-range scans below must report
        // exactly that index.
        __host2 = __host;
        if (__has_match)
            __host2[__pos] = 0;
        __dt.update_data(__host.data());
        __dt2.update_data(__host2.data());

        // Forward tag: the first match.
        EXPECT_TRUE(oneapi::dpl::find_if(__exec_find_if, __d, __d + __n, __is_one) == __d + __pos,
                    "wrong index from find_if");
        // Backward tag: find_end over a one-element needle returns the last match. Its brick declines the
        // wide scan, so this covers the narrow path.
        if (__n > 1)
        {
            auto __expected = __has_match ? __d + __decoy : __d + __n;
            EXPECT_TRUE(oneapi::dpl::find_end(__exec_find_end, __d, __d + __n, __nd, __nd + 1) == __expected,
                        "wrong index from find_end");
        }
        // Or tag: presence only.
        EXPECT_TRUE(oneapi::dpl::any_of(__exec_any_of, __d, __d + __n, __is_one) == __has_match,
                    "wrong result from any_of");
        EXPECT_TRUE(oneapi::dpl::none_of(__exec_none_of, __d, __d + __n, __is_one) == !__has_match,
                    "wrong result from none_of");
        // Two ranges over one allocation, so the scan reads two elements per index from aliased views.
        EXPECT_TRUE(oneapi::dpl::mismatch(__exec_mismatch_self, __d, __d + __n, __d).first == __d + __n,
                    "wrong result from mismatch of a range with itself");
        // Two distinct ranges, so the returned index is checked and not only the end sentinel.
        EXPECT_TRUE(oneapi::dpl::mismatch(__exec_mismatch, __d, __d + __n, __d2).first ==
                        (__has_match ? __d + __pos : __d + __n),
                    "wrong index from mismatch of two 4-byte ranges");
        EXPECT_TRUE(oneapi::dpl::equal(__exec_equal, __d, __d + __n, __d2) == !__has_match,
                    "wrong result from equal of two 4-byte ranges");
    }
}

// Two 8-byte ranges. The width gate routes them to the narrow scan.
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
    TestUtils::usm_data_transfer<sycl::usm::alloc::device, _T> __dt1(__q, __host.begin(), __n);
    TestUtils::usm_data_transfer<sycl::usm::alloc::device, _T> __dt2(__q, __n);
    _T* __d1 = __dt1.get_data();
    _T* __d2 = __dt2.get_data();

    const std::size_t __step = match_position_step(__n);

    for (std::size_t __pos = 0; __pos <= __n; __pos += __step)
    {
        std::fill(__host.begin(), __host.end(), _T(1));
        const bool __differs = __pos < __n;
        if (__differs)
            __host[__pos] = _T(2);
        __dt2.update_data(__host.data());

        EXPECT_TRUE(oneapi::dpl::mismatch(__exec_mismatch, __d1, __d1 + __n, __d2).first ==
                        (__differs ? __d1 + __pos : __d1 + __n),
                    "wrong index from mismatch of two 8-byte ranges");
        EXPECT_TRUE(oneapi::dpl::equal(__exec_equal, __d1, __d1 + __n, __d2) == !__differs,
                    "wrong result from equal of two 8-byte ranges");
    }
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
    TestUtils::usm_data_transfer<sycl::usm::alloc::device, _T> __dt1(__q, __n);
    // The second range stays all-zero, so one match in the first serves every tag below.
    TestUtils::usm_data_transfer<sycl::usm::alloc::device, _T> __dt2(__q, __host.begin(), __n);
    _T* __d1 = __dt1.get_data();
    _T* __d2 = __dt2.get_data();

    const std::size_t __step = match_position_step(__n);

    for (std::size_t __pos = 0; __pos <= __n; __pos += __step)
    {
        std::fill(__host.begin(), __host.end(), _T(0));
        const bool __has_match = __pos < __n;
        if (__has_match)
            __host[__pos] = _T(1);
        __dt1.update_data(__host.data());

        // Forward tag: the first match.
        EXPECT_TRUE(oneapi::dpl::find_if(__exec_find_if, __d1, __d1 + __n, __is_one) == __d1 + __pos,
                    "wrong index from find_if over 2-byte elements");
        // Or tag: presence only.
        EXPECT_TRUE(oneapi::dpl::any_of(__exec_any_of, __d1, __d1 + __n, __is_one) == __has_match,
                    "wrong result from any_of over 2-byte elements");
        EXPECT_TRUE(oneapi::dpl::none_of(__exec_none_of, __d1, __d1 + __n, __is_one) == !__has_match,
                    "wrong result from none_of over 2-byte elements");
        EXPECT_TRUE(oneapi::dpl::mismatch(__exec_mismatch, __d1, __d1 + __n, __d2).first ==
                        (__has_match ? __d1 + __pos : __d1 + __n),
                    "wrong index from mismatch of two 2-byte ranges");
        EXPECT_TRUE(oneapi::dpl::equal(__exec_equal, __d1, __d1 + __n, __d2) == !__has_match,
                    "wrong result from equal of two 2-byte ranges");
    }
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
