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

    // The read_archetype family: the read-only algorithms which are parameterized by a callable only.
    // Covers for_each, find_if, find_if_not, find_last_if, find_last_if_not, any_of, all_of, none_of,
    // is_partitioned, count_if, min_element, max_element, minmax_element, is_sorted, is_sorted_until and
    // adjacent_find, first with const callables and then with callables taking non-const references.

    // read_archetype is neither copyable, movable, default constructible nor comparable; the only
    // operations available are the ones the callables of the algorithm provide.
    run_algo_all_policies<read_archetype, read_archetype_dc, __LINE__>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::for_each(std::forward<decltype(policy)>(policy), view, read_unary_fun{});
        },
        [](auto&& view, auto res) { return res == std::ranges::begin(view) + std::ranges::size(view); }, "for_each");

    run_algo_all_policies<read_archetype, read_archetype_dc, __LINE__>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::find_if(std::forward<decltype(policy)>(policy), view, read_unary_pred{});
        },
        [](auto&& view, auto res) { return res == std::ranges::begin(view); }, "find_if");

    run_algo_all_policies<read_archetype, read_archetype_dc, __LINE__>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::find_if_not(std::forward<decltype(policy)>(policy), view, read_unary_pred{});
        },
        [](auto&& view, auto res) { return res == std::ranges::begin(view) + 1; }, "find_if_not");

    // The last element whose value is divisible by three, and the last one whose value is not.
    run_algo_all_policies<read_archetype, read_archetype_dc, __LINE__>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::find_last_if(std::forward<decltype(policy)>(policy), view, read_unary_pred{});
        },
        [](auto&& view, auto res) {
            auto __n = (int)std::ranges::size(view);
            return std::ranges::begin(res) == std::ranges::begin(view) + (__n - 1) / 3 * 3;
        },
        "find_last_if");

    run_algo_all_policies<read_archetype, read_archetype_dc, __LINE__>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::find_last_if_not(std::forward<decltype(policy)>(policy), view, read_unary_pred{});
        },
        [](auto&& view, auto res) {
            auto __n = (int)std::ranges::size(view);
            return std::ranges::begin(res) == std::ranges::begin(view) + ((__n - 1) % 3 == 0 ? __n - 2 : __n - 1);
        },
        "find_last_if_not");

    run_algo_all_policies<read_archetype, read_archetype_dc, __LINE__>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::any_of(std::forward<decltype(policy)>(policy), view, read_unary_pred{});
        },
        [](auto&&, bool res) { return res; }, "any_of");

    run_algo_all_policies<read_archetype, read_archetype_dc, __LINE__>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::all_of(std::forward<decltype(policy)>(policy), view, read_unary_pred{});
        },
        [](auto&&, bool res) { return !res; }, "all_of");

    run_algo_all_policies<read_archetype, read_archetype_dc, __LINE__>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::none_of(std::forward<decltype(policy)>(policy), view, read_unary_pred{});
        },
        [](auto&&, bool res) { return !res; }, "none_of");

    // The predicate holds for 0, fails for 1 and holds again for 3, so the range is not partitioned.
    run_algo_all_policies<read_archetype, read_archetype_dc, __LINE__>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::is_partitioned(std::forward<decltype(policy)>(policy), view, read_unary_pred{});
        },
        [](auto&&, bool res) { return !res; }, "is_partitioned");

    run_algo_all_policies<read_archetype, read_archetype_dc, __LINE__>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::count_if(std::forward<decltype(policy)>(policy), view, read_unary_pred{});
        },
        [](auto&& view, auto res) { return res == (std::ranges::range_difference_t<decltype(view)>)
                                                      ((std::ranges::size(view) + 2) / 3); }, "count_if");

    // The projection returns an unrelated prvalue type, so the predicate can only ever be applied to
    // the projected value.
    run_algo_all_policies<read_archetype, read_archetype_dc, __LINE__>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::find_if(std::forward<decltype(policy)>(policy), view, read_proj_pred{}, read_proj{});
        },
        [](auto&& view, auto res) { return res == std::ranges::begin(view); }, "find_if with proj");

    run_algo_all_policies<read_archetype, read_archetype_dc, __LINE__>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::count_if(std::forward<decltype(policy)>(policy), view, read_proj_pred{}, read_proj{});
        },
        [](auto&& view, auto res) { return res == (std::ranges::range_difference_t<decltype(view)>)
                                                      ((std::ranges::size(view) + 2) / 3); }, "count_if with proj");

    // min_element/max_element/minmax_element only require std::indirect_strict_weak_order on the
    // projected iterator, so the element type stays non-copyable and non-default-constructible: both
    // backends carry an index and dereference the iterator for the comparison instead of storing the
    // element by value.
    run_algo_all_policies<read_archetype, read_archetype_dc, __LINE__>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::min_element(std::forward<decltype(policy)>(policy), view, read_comp{});
        },
        [](auto&& view, auto res) { return res == std::ranges::begin(view); }, "min_element");

    run_algo_all_policies<read_archetype, read_archetype_dc, __LINE__>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::max_element(std::forward<decltype(policy)>(policy), view, read_comp{});
        },
        [](auto&& view, auto res) { return res == std::ranges::begin(view) + std::ranges::size(view) - 1; },
        "max_element");

    run_algo_all_policies<read_archetype, read_archetype_dc, __LINE__>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::minmax_element(std::forward<decltype(policy)>(policy), view, read_comp{});
        },
        [](auto&& view, auto res) {
            return res.min == std::ranges::begin(view) &&
                   res.max == std::ranges::begin(view) + std::ranges::size(view) - 1;
        },
        "minmax_element");

    run_algo_all_policies<read_archetype, read_archetype_dc, __LINE__>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::is_sorted(std::forward<decltype(policy)>(policy), view, read_comp{});
        },
        [](auto&&, bool res) { return res; }, "is_sorted");

    // The whole range is sorted, so the scan stops at its end.
    run_algo_all_policies<read_archetype, read_archetype_dc, __LINE__>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::is_sorted_until(std::forward<decltype(policy)>(policy), view, read_comp{});
        },
        [](auto&& view, auto res) { return res == std::ranges::begin(view) + std::ranges::size(view); },
        "is_sorted_until");

    run_algo_all_policies<read_archetype, read_archetype_dc, __LINE__>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::adjacent_find(std::forward<decltype(policy)>(policy), view, read_binary_pred{});
        },
        [](auto&& view, auto res) { return res == std::ranges::begin(view) + std::ranges::size(view); },
        "adjacent_find");

    //----------------------------------------------------------------------------------------------
    // The same algorithms with callables taking their arguments by non-const reference.
    //----------------------------------------------------------------------------------------------
    run_algo_all_policies<read_archetype, read_archetype_dc, __LINE__>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::for_each(std::forward<decltype(policy)>(policy), view, read_unary_fun_mut{});
        },
        [](auto&& view, auto res) { return res == std::ranges::begin(view) + std::ranges::size(view); },
        "for_each, non-const callable");

    run_algo_all_policies<read_archetype, read_archetype_dc, __LINE__>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::find_if(std::forward<decltype(policy)>(policy), view, read_unary_pred_mut{});
        },
        [](auto&& view, auto res) { return res == std::ranges::begin(view); }, "find_if, non-const callable");

    run_algo_all_policies<read_archetype, read_archetype_dc, __LINE__>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::find_if_not(std::forward<decltype(policy)>(policy), view, read_unary_pred_mut{});
        },
        [](auto&& view, auto res) { return res == std::ranges::begin(view) + 1; }, "find_if_not, non-const callable");

    // The last element whose value is divisible by three, and the last one whose value is not.
    run_algo_all_policies<read_archetype, read_archetype_dc, __LINE__>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::find_last_if(std::forward<decltype(policy)>(policy), view, read_unary_pred_mut{});
        },
        [](auto&& view, auto res) {
            auto __n = (int)std::ranges::size(view);
            return std::ranges::begin(res) == std::ranges::begin(view) + (__n - 1) / 3 * 3;
        },
        "find_last_if, non-const callable");

    run_algo_all_policies<read_archetype, read_archetype_dc, __LINE__>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::find_last_if_not(std::forward<decltype(policy)>(policy), view, read_unary_pred_mut{});
        },
        [](auto&& view, auto res) {
            auto __n = (int)std::ranges::size(view);
            return std::ranges::begin(res) == std::ranges::begin(view) + ((__n - 1) % 3 == 0 ? __n - 2 : __n - 1);
        },
        "find_last_if_not, non-const callable");

    run_algo_all_policies<read_archetype, read_archetype_dc, __LINE__>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::any_of(std::forward<decltype(policy)>(policy), view, read_unary_pred_mut{});
        },
        [](auto&&, bool res) { return res; }, "any_of, non-const callable");

    // Every third element satisfies the predicate, so the range is neither all nor none of it, and it
    // is not partitioned either.
    run_algo_all_policies<read_archetype, read_archetype_dc, __LINE__>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::all_of(std::forward<decltype(policy)>(policy), view, read_unary_pred_mut{});
        },
        [](auto&&, bool res) { return !res; }, "all_of, non-const callable");

    run_algo_all_policies<read_archetype, read_archetype_dc, __LINE__>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::none_of(std::forward<decltype(policy)>(policy), view, read_unary_pred_mut{});
        },
        [](auto&&, bool res) { return !res; }, "none_of, non-const callable");

    // KSATODO: std::indirect_unary_predicate only requires the predicate to be invocable with
    // iter_reference_t<_It>, a non-const lvalue here, but the device path applies it to a const
    // lvalue, so the call does not compile:
    //  - algorithm_impl_hetero.h:1078,1080 - __pattern_is_partitioned_transform_fn::operator() is
    //    const and takes the accessor by value, so __acc[__gidx] yields a const reference which is
    //    passed straight into the predicate.
    run_algo_all_policies<read_archetype, read_archetype_dc, __LINE__, true,
                          !_TEST_CPP20_RANGES_BROKEN_REQUIRES_IS_PARTITIONED_HETERO>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::is_partitioned(std::forward<decltype(policy)>(policy), view, read_unary_pred_mut{});
        },
        [](auto&&, bool res) { return !res; }, "is_partitioned, non-const callable");

    run_algo_all_policies<read_archetype, read_archetype_dc, __LINE__>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::count_if(std::forward<decltype(policy)>(policy), view, read_unary_pred_mut{});
        },
        [](auto&& view, auto res) {
            return res == (std::ranges::range_difference_t<decltype(view)>)((std::ranges::size(view) + 2) / 3);
        },
        "count_if, non-const callable");

    // The projection takes the element by non-const reference; the predicate sees its prvalue result.
    run_algo_all_policies<read_archetype, read_archetype_dc, __LINE__>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::find_if(std::forward<decltype(policy)>(policy), view, read_proj_pred{}, read_proj_mut{});
        },
        [](auto&& view, auto res) { return res == std::ranges::begin(view); }, "find_if, non-const projection");

    run_algo_all_policies<read_archetype, read_archetype_dc, __LINE__>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::count_if(std::forward<decltype(policy)>(policy), view, read_proj_pred{},
                                        read_proj_mut{});
        },
        [](auto&& view, auto res) {
            return res == (std::ranges::range_difference_t<decltype(view)>)((std::ranges::size(view) + 2) / 3);
        },
        "count_if, non-const projection");

    run_algo_all_policies<read_archetype, read_archetype_dc, __LINE__>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::adjacent_find(std::forward<decltype(policy)>(policy), view, read_binary_pred_mut{});
        },
        [](auto&& view, auto res) { return res == std::ranges::begin(view) + std::ranges::size(view); },
        "adjacent_find, non-const callable");

    run_algo_all_policies<read_archetype, read_archetype_dc, __LINE__>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::is_sorted(std::forward<decltype(policy)>(policy), view, read_comp_mut{});
        },
        [](auto&&, bool res) { return res; }, "is_sorted, non-const comparator");

    // The whole range is sorted, so the scan stops at its end.
    run_algo_all_policies<read_archetype, read_archetype_dc, __LINE__>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::is_sorted_until(std::forward<decltype(policy)>(policy), view, read_comp_mut{});
        },
        [](auto&& view, auto res) { return res == std::ranges::begin(view) + std::ranges::size(view); },
        "is_sorted_until, non-const comparator");

    run_algo_all_policies<read_archetype, read_archetype_dc, __LINE__>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::min_element(std::forward<decltype(policy)>(policy), view, read_comp_mut{});
        },
        [](auto&& view, auto res) { return res == std::ranges::begin(view); }, "min_element, non-const comparator");

    run_algo_all_policies<read_archetype, read_archetype_dc, __LINE__>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::max_element(std::forward<decltype(policy)>(policy), view, read_comp_mut{});
        },
        [](auto&& view, auto res) { return res == std::ranges::begin(view) + std::ranges::size(view) - 1; },
        "max_element, non-const comparator");

    run_algo_all_policies<read_archetype, read_archetype_dc, __LINE__>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::minmax_element(std::forward<decltype(policy)>(policy), view, read_comp_mut{});
        },
        [](auto&& view, auto res) {
            return res.min == std::ranges::begin(view) &&
                   res.max == std::ranges::begin(view) + std::ranges::size(view) - 1;
        },
        "minmax_element, non-const comparator");

#endif //_ENABLE_STD_RANGES_TESTING

    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
