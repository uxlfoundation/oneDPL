// -*- C++ -*-
//===----------------------------------------------------------------------===//
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

#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>

#include "support/test_config.h"
#include "support/test_macros.h"
#include "support/utils.h"

#if _ENABLE_STD_RANGES_TESTING
#include "std_ranges_archetypes.h"
#include "std_ranges_algo_archetypes_test.h"
#endif //_ENABLE_STD_RANGES_TESTING

int
main()
{
#if _ENABLE_STD_RANGES_TESTING
    using namespace test_std_ranges;
    using namespace test_std_ranges::archetypes;
    namespace dpl_ranges = oneapi::dpl::ranges;

    // The permutable archetype family, i.e. the algorithms constrained by std::permutable or by
    // std::sortable: reverse, remove_if, unique, partition, sort, stable_sort, is_sorted, and, with a
    // non-const comparator only, partial_sort, nth_element and inplace_merge. The first section calls
    // them with const callables, the second one with callables taking their arguments by reference.

    //----------------------------------------------------------------------------------------------
    // Const callables.
    //----------------------------------------------------------------------------------------------
    // permutable_archetype is movable but not copyable, so it is not device copyable either: the
    // host policies are the only ones its constraints allow.
    run_algo_all_policies<permutable_archetype, permutable_archetype_dc, __LINE__>(
        [](auto&& policy, auto&& view) { return dpl_ranges::reverse(std::forward<decltype(policy)>(policy), view); },
        [](auto&& view, auto) {
            const auto n = std::ranges::size(view);
            return std::ranges::begin(view)[0].val == (int)n - 1 && std::ranges::begin(view)[n - 1].val == 0;
        },
        "reverse");

    // The storage is filled with 0, 1, 2, ... so every third element is removed. The returned
    // subrange is the tail holding the removed elements.
    run_algo_all_policies<permutable_archetype, permutable_archetype_dc, __LINE__>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::remove_if(std::forward<decltype(policy)>(policy), view, permutable_pred{});
        },
        [](auto&& view, auto res) {
            const auto n = std::ranges::size(view);
            return std::ranges::size(res) == (n + 2) / 3;
        },
        "remove_if");

    // All the elements are unique, so nothing is dropped.
    run_algo_all_policies<permutable_archetype, permutable_archetype_dc, __LINE__>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::unique(std::forward<decltype(policy)>(policy), view, permutable_equiv{});
        },
        [](auto&& view, auto res) { return std::ranges::size(res) == 0; }, "unique");

    // partition returns the tail of the elements which do not satisfy the predicate.
    run_algo_all_policies<permutable_archetype, permutable_archetype_dc, __LINE__>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::partition(std::forward<decltype(policy)>(policy), view, permutable_pred{});
        },
        [](auto&& view, auto res) {
            return std::ranges::size(res) == std::ranges::size(view) - (std::ranges::size(view) + 2) / 3;
        },
        "partition");

    // The storage of the harness is filled in ascending order, so sorting it keeps it as it is: what
    // these two cases check is that the call compiles and leaves the range intact, not the ordering.
    run_algo_all_policies<permutable_archetype, permutable_archetype_dc, __LINE__>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::sort(std::forward<decltype(policy)>(policy), view, permutable_comp{});
        },
        [](auto&& view, auto) {
            return std::ranges::begin(view)[0].val == 0 &&
                   std::ranges::begin(view)[std::ranges::size(view) - 1].val == (int)std::ranges::size(view) - 1;
        },
        "sort");

    run_algo_all_policies<permutable_archetype, permutable_archetype_dc, __LINE__>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::stable_sort(std::forward<decltype(policy)>(policy), view, permutable_comp{});
        },
        [](auto&& view, auto) {
            return std::ranges::begin(view)[0].val == 0 &&
                   std::ranges::begin(view)[std::ranges::size(view) - 1].val == (int)std::ranges::size(view) - 1;
        },
        "stable_sort");

    run_algo_all_policies<permutable_archetype, permutable_archetype_dc, __LINE__>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::is_sorted(std::forward<decltype(policy)>(policy), view, permutable_comp{});
        },
        [](auto&&, auto res) { return res; }, "is_sorted");

    //----------------------------------------------------------------------------------------------
    // Callables taking their arguments by non-const reference. The element of a permutable range is
    // mutable by definition, so its predicate and its comparator only ever see a non-const lvalue and
    // are not required to accept a const one.
    //----------------------------------------------------------------------------------------------
    // Every third element is removed; the returned subrange is the tail holding the removed elements.
    run_algo_all_policies<permutable_archetype, permutable_archetype_dc, __LINE__>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::remove_if(std::forward<decltype(policy)>(policy), view, permutable_pred_mut{});
        },
        [](auto&& view, auto res) { return std::ranges::size(res) == (std::ranges::size(view) + 2) / 3; },
        "remove_if, non-const callable");

    // All the elements are unique, so nothing is dropped.
    run_algo_all_policies<permutable_archetype, permutable_archetype_dc, __LINE__>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::unique(std::forward<decltype(policy)>(policy), view, permutable_equiv_mut{});
        },
        [](auto&&, auto res) { return std::ranges::size(res) == 0; }, "unique, non-const callable");

    // partition returns the tail of the elements which do not satisfy the predicate.
    //
    // KSATODO: the device path of partition applies the predicate to a const lvalue, which
    // std::indirect_unary_predicate over a permutable iterator does not ask for, so it does not
    // compile:
    //  - unseq_backend_sycl.h:122 - walk_n::operator() is const and calls __f(__rngs[__idx]...) on
    //    the const range members of single_match_pred_by_idx.
    run_algo_all_policies<permutable_archetype, permutable_archetype_dc, __LINE__, true,
                          !_TEST_CPP20_RANGES_BROKEN_REQUIRES_PARTITION_HETERO>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::partition(std::forward<decltype(policy)>(policy), view, permutable_pred_mut{});
        },
        [](auto&& view, auto res) {
            return std::ranges::size(res) == std::ranges::size(view) - (std::ranges::size(view) + 2) / 3;
        },
        "partition, non-const callable");

    // The very same comparator as the one the sorting algorithms below are called with.
    run_algo_all_policies<permutable_archetype, permutable_archetype_dc, __LINE__>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::is_sorted(std::forward<decltype(policy)>(policy), view, permutable_comp_mut{});
        },
        [](auto&&, bool res) { return res; }, "is_sorted of a permutable range, non-const comparator");

    // KSATODO: std::sortable<_It, _Comp> only requires the comparator to be invocable with
    // iter_reference_t<_It>, which is a non-const lvalue for archetype_view, so a comparator taking
    // its arguments by non-const reference is enough. The parallel host merge sort compares against a
    // const lvalue instead, so par and par_unseq do not compile:
    //  - parallel_backend_tbb.h:1037 - std::lower_bound(..., _M_comp) passes the const lvalue _Val
    //    of the merge split point to the comparator;
    //  - utils.h:203 - __binary_op::operator() forwards that const lvalue into std::invoke.
    // seq and unseq keep the element non-const all the way down.
    run_algo_all_policies<permutable_archetype, permutable_archetype_dc, __LINE__,
                          !_TEST_CPP20_RANGES_BROKEN_REQUIRES_SORT_HOST>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::sort(std::forward<decltype(policy)>(policy), view, permutable_comp_mut{});
        },
        [](auto&& view, auto) {
            return std::ranges::begin(view)[0].val == 0 &&
                   std::ranges::begin(view)[std::ranges::size(view) - 1].val == (int)std::ranges::size(view) - 1;
        },
        "sort, non-const comparator");

    // KSATODO: stable_sort shares the merge sort of the parallel host policies with sort, so it is
    // broken for par and par_unseq in exactly the same way, see the note above.
    run_algo_all_policies<permutable_archetype, permutable_archetype_dc, __LINE__,
                          !_TEST_CPP20_RANGES_BROKEN_REQUIRES_STABLE_SORT_HOST>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::stable_sort(std::forward<decltype(policy)>(policy), view, permutable_comp_mut{});
        },
        [](auto&& view, auto) {
            return std::ranges::begin(view)[0].val == 0 &&
                   std::ranges::begin(view)[std::ranges::size(view) - 1].val == (int)std::ranges::size(view) - 1;
        },
        "stable_sort, non-const comparator");

    // KSATODO: partial_sort shares the parallel merge sort with sort, so the parallel host policies
    // hand a const lvalue to the comparator here as well and par / par_unseq do not compile:
    //  - parallel_backend_tbb.h:1023,1026,1034 - __merge_func::split_merging passes *(_M_x_beg + __ym)
    //    to std::upper_bound / std::lower_bound, which compares against their const lvalue parameter;
    //  - utils.h:203 - __binary_op::operator() forwards that const lvalue into std::invoke.
    // seq and unseq keep the element non-const all the way down.
    // The range is ascending already, so the first ten elements are 0 ... 9 afterwards.
    run_algo_all_policies<permutable_archetype, permutable_archetype_dc, __LINE__,
                          !_TEST_CPP20_RANGES_BROKEN_REQUIRES_PARTIAL_SORT_HOST>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::partial_sort(std::forward<decltype(policy)>(policy), view, std::ranges::begin(view) + 10,
                                            permutable_comp_mut{});
        },
        [](auto&& view, auto) { return std::ranges::begin(view)[0].val == 0 && std::ranges::begin(view)[9].val == 9; },
        "partial_sort, non-const comparator");

    // KSATODO: std::sortable only requires the comparator to be invocable with the non-const
    // reference of the element, but the parallel path of nth_element compares against a const lvalue,
    // so par and par_unseq do not compile:
    //  - algorithm_impl.h:2841 - the partition predicate of the quickselect loop takes const _Tp& and
    //    passes it into std::invoke(__comp, __x, *__first).
    // Taking the element by reference in that lambda is enough to fix it; seq and unseq are fine.
    run_algo_all_policies<permutable_archetype, permutable_archetype_dc, __LINE__,
                          !_TEST_CPP20_RANGES_BROKEN_REQUIRES_NTH_ELEMENT_HOST>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::nth_element(std::forward<decltype(policy)>(policy), view, std::ranges::begin(view) + 10,
                                           permutable_comp_mut{});
        },
        [](auto&& view, auto) { return std::ranges::begin(view)[10].val == 10; }, "nth_element, non-const comparator");

    // KSATODO: inplace_merge does not compile with any host policy, for two independent reasons:
    //  - algorithm_ranges_impl.h:848 - the serial path returns __end(__r), i.e. the sentinel of the
    //    range, while the declared return type is std::ranges::borrowed_iterator_t<_R>. For a range
    //    which is not a common_range the two types differ, so seq already fails to compile. This one
    //    is independent of the comparator and hits any user range with a distinct sentinel type;
    //  - the const lvalue of the merge split point is handed to the comparator, which std::sortable
    //    never asks for: std::inplace_merge compares against its const value parameter for unseq, and
    //    parallel_backend_tbb.h:1240,1245 does the same through std::upper_bound / std::lower_bound
    //    for par and par_unseq.
    // Both halves of the ascending range are sorted, so merging them keeps it as it is.
    //
    // KSATODO: the device path of inplace_merge compares two const lvalues, which std::sortable does
    // not ask for, so the call does not compile:
    //  - parallel_backend_sycl_merge.h:128-133 - the lambda of __find_start_point captures __rng1 and
    //    __rng2 and subscripts them as const, and both results go into the comparator.
    run_algo_all_policies<permutable_archetype, permutable_archetype_dc, __LINE__,
                          !_TEST_CPP20_RANGES_BROKEN_REQUIRES_INPLACE_MERGE_HOST,
                          !_TEST_CPP20_RANGES_BROKEN_REQUIRES_INPLACE_MERGE_HETERO>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::inplace_merge(std::forward<decltype(policy)>(policy), view,
                                             std::ranges::begin(view) + std::ranges::size(view) / 2,
                                             permutable_comp_mut{});
        },
        [](auto&& view, auto) {
            return std::ranges::begin(view)[0].val == 0 &&
                   std::ranges::begin(view)[std::ranges::size(view) - 1].val == (int)std::ranges::size(view) - 1;
        },
        "inplace_merge, non-const comparator");

#endif //_ENABLE_STD_RANGES_TESTING

    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
