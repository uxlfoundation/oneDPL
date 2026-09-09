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

    // The storage is filled with the values 0, 1, 2, ...
    constexpr int searched = 3;

    //----------------------------------------------------------------------------------------------
    // Read-only algorithms with a callable taking the element by non-const reference.
    //----------------------------------------------------------------------------------------------
    run_algo_host_policies<read_archetype>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::for_each(std::forward<decltype(policy)>(policy), view, read_unary_fun_mut{});
        },
        [](auto&& view, auto res) { return res == std::ranges::begin(view) + std::ranges::size(view); },
        "for_each, non-const callable");

#if TEST_DPCPP_BACKEND_PRESENT
    run_algo_hetero_policies<read_archetype_dc, 0>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::for_each(std::forward<decltype(policy)>(policy), view, read_unary_fun_mut{});
        },
        [](auto&& view, auto res) { return res == std::ranges::begin(view) + std::ranges::size(view); },
        "for_each, non-const callable");
#endif // TEST_DPCPP_BACKEND_PRESENT

    run_algo_host_policies<read_archetype>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::find_if(std::forward<decltype(policy)>(policy), view, read_unary_pred_mut{});
        },
        [](auto&& view, auto res) { return res == std::ranges::begin(view); }, "find_if, non-const callable");

#if TEST_DPCPP_BACKEND_PRESENT
    run_algo_hetero_policies<read_archetype_dc, 1>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::find_if(std::forward<decltype(policy)>(policy), view, read_unary_pred_mut{});
        },
        [](auto&& view, auto res) { return res == std::ranges::begin(view); }, "find_if, non-const callable");
#endif // TEST_DPCPP_BACKEND_PRESENT

    run_algo_host_policies<read_archetype>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::find_if_not(std::forward<decltype(policy)>(policy), view, read_unary_pred_mut{});
        },
        [](auto&& view, auto res) { return res == std::ranges::begin(view) + 1; }, "find_if_not, non-const callable");

#if TEST_DPCPP_BACKEND_PRESENT
    run_algo_hetero_policies<read_archetype_dc, 26>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::find_if_not(std::forward<decltype(policy)>(policy), view, read_unary_pred_mut{});
        },
        [](auto&& view, auto res) { return res == std::ranges::begin(view) + 1; }, "find_if_not, non-const callable");
#endif // TEST_DPCPP_BACKEND_PRESENT

    // The last element whose value is divisible by three, and the last one whose value is not.
    run_algo_host_policies<read_archetype>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::find_last_if(std::forward<decltype(policy)>(policy), view, read_unary_pred_mut{});
        },
        [](auto&& view, auto res) {
            auto __n = (int)std::ranges::size(view);
            return std::ranges::begin(res) == std::ranges::begin(view) + (__n - 1) / 3 * 3;
        },
        "find_last_if, non-const callable");

#if TEST_DPCPP_BACKEND_PRESENT
    run_algo_hetero_policies<read_archetype_dc, 27>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::find_last_if(std::forward<decltype(policy)>(policy), view, read_unary_pred_mut{});
        },
        [](auto&& view, auto res) {
            auto __n = (int)std::ranges::size(view);
            return std::ranges::begin(res) == std::ranges::begin(view) + (__n - 1) / 3 * 3;
        },
        "find_last_if, non-const callable");
#endif // TEST_DPCPP_BACKEND_PRESENT

    run_algo_host_policies<read_archetype>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::find_last_if_not(std::forward<decltype(policy)>(policy), view, read_unary_pred_mut{});
        },
        [](auto&& view, auto res) {
            auto __n = (int)std::ranges::size(view);
            return std::ranges::begin(res) == std::ranges::begin(view) + ((__n - 1) % 3 == 0 ? __n - 2 : __n - 1);
        },
        "find_last_if_not, non-const callable");

#if TEST_DPCPP_BACKEND_PRESENT
    run_algo_hetero_policies<read_archetype_dc, 28>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::find_last_if_not(std::forward<decltype(policy)>(policy), view, read_unary_pred_mut{});
        },
        [](auto&& view, auto res) {
            auto __n = (int)std::ranges::size(view);
            return std::ranges::begin(res) == std::ranges::begin(view) + ((__n - 1) % 3 == 0 ? __n - 2 : __n - 1);
        },
        "find_last_if_not, non-const callable");
#endif // TEST_DPCPP_BACKEND_PRESENT

    run_algo_host_policies<read_archetype>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::any_of(std::forward<decltype(policy)>(policy), view, read_unary_pred_mut{});
        },
        [](auto&&, bool res) { return res; }, "any_of, non-const callable");

#if TEST_DPCPP_BACKEND_PRESENT
    run_algo_hetero_policies<read_archetype_dc, 2>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::any_of(std::forward<decltype(policy)>(policy), view, read_unary_pred_mut{});
        },
        [](auto&&, bool res) { return res; }, "any_of, non-const callable");
#endif // TEST_DPCPP_BACKEND_PRESENT

    // Every third element satisfies the predicate, so the range is neither all nor none of it, and it
    // is not partitioned either.
    run_algo_host_policies<read_archetype>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::all_of(std::forward<decltype(policy)>(policy), view, read_unary_pred_mut{});
        },
        [](auto&&, bool res) { return !res; }, "all_of, non-const callable");

#if TEST_DPCPP_BACKEND_PRESENT
    run_algo_hetero_policies<read_archetype_dc, 29>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::all_of(std::forward<decltype(policy)>(policy), view, read_unary_pred_mut{});
        },
        [](auto&&, bool res) { return !res; }, "all_of, non-const callable");
#endif // TEST_DPCPP_BACKEND_PRESENT

    run_algo_host_policies<read_archetype>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::none_of(std::forward<decltype(policy)>(policy), view, read_unary_pred_mut{});
        },
        [](auto&&, bool res) { return !res; }, "none_of, non-const callable");

#if TEST_DPCPP_BACKEND_PRESENT
    run_algo_hetero_policies<read_archetype_dc, 30>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::none_of(std::forward<decltype(policy)>(policy), view, read_unary_pred_mut{});
        },
        [](auto&&, bool res) { return !res; }, "none_of, non-const callable");
#endif // TEST_DPCPP_BACKEND_PRESENT

    run_algo_host_policies<read_archetype>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::is_partitioned(std::forward<decltype(policy)>(policy), view, read_unary_pred_mut{});
        },
        [](auto&&, bool res) { return !res; }, "is_partitioned, non-const callable");

    // KSATODO: std::indirect_unary_predicate only requires the predicate to be invocable with
    // iter_reference_t<_It>, a non-const lvalue here, but the device path applies it to a const
    // lvalue, so the call does not compile:
    //  - algorithm_impl_hetero.h:1078,1080 - __pattern_is_partitioned_transform_fn::operator() is
    //    const and takes the accessor by value, so __acc[__gidx] yields a const reference which is
    //    passed straight into the predicate.
#if TEST_DPCPP_BACKEND_PRESENT
#if !_TEST_CPP20_RANGES_BROKEN_REQUIRES_IS_PARTITIONED_HETERO
    run_algo_hetero_policies<read_archetype_dc, 31>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::is_partitioned(std::forward<decltype(policy)>(policy), view, read_unary_pred_mut{});
        },
        [](auto&&, bool res) { return !res; }, "is_partitioned, non-const callable");
#endif // !_TEST_CPP20_RANGES_BROKEN_REQUIRES_IS_PARTITIONED_HETERO
#endif // TEST_DPCPP_BACKEND_PRESENT

    run_algo_host_policies<read_archetype>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::count_if(std::forward<decltype(policy)>(policy), view, read_unary_pred_mut{});
        },
        [](auto&& view, auto res) {
            return res == (std::ranges::range_difference_t<decltype(view)>)((std::ranges::size(view) + 2) / 3);
        },
        "count_if, non-const callable");

#if TEST_DPCPP_BACKEND_PRESENT
    run_algo_hetero_policies<read_archetype_dc, 3>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::count_if(std::forward<decltype(policy)>(policy), view, read_unary_pred_mut{});
        },
        [](auto&& view, auto res) {
            return res == (std::ranges::range_difference_t<decltype(view)>)((std::ranges::size(view) + 2) / 3);
        },
        "count_if, non-const callable");
#endif // TEST_DPCPP_BACKEND_PRESENT

    // The projection takes the element by non-const reference; the predicate sees its prvalue result.
    run_algo_host_policies<read_archetype>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::find_if(std::forward<decltype(policy)>(policy), view, read_proj_pred{}, read_proj_mut{});
        },
        [](auto&& view, auto res) { return res == std::ranges::begin(view); }, "find_if, non-const projection");

#if TEST_DPCPP_BACKEND_PRESENT
    run_algo_hetero_policies<read_archetype_dc, 32>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::find_if(std::forward<decltype(policy)>(policy), view, read_proj_pred{}, read_proj_mut{});
        },
        [](auto&& view, auto res) { return res == std::ranges::begin(view); }, "find_if, non-const projection");
#endif // TEST_DPCPP_BACKEND_PRESENT

    run_algo_host_policies<read_archetype>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::count_if(std::forward<decltype(policy)>(policy), view, read_proj_pred{},
                                        read_proj_mut{});
        },
        [](auto&& view, auto res) {
            return res == (std::ranges::range_difference_t<decltype(view)>)((std::ranges::size(view) + 2) / 3);
        },
        "count_if, non-const projection");

#if TEST_DPCPP_BACKEND_PRESENT
    run_algo_hetero_policies<read_archetype_dc, 4>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::count_if(std::forward<decltype(policy)>(policy), view, read_proj_pred{},
                                        read_proj_mut{});
        },
        [](auto&& view, auto res) {
            return res == (std::ranges::range_difference_t<decltype(view)>)((std::ranges::size(view) + 2) / 3);
        },
        "count_if, non-const projection");
#endif // TEST_DPCPP_BACKEND_PRESENT

    run_algo_host_policies<read_archetype>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::adjacent_find(std::forward<decltype(policy)>(policy), view, read_binary_pred_mut{});
        },
        [](auto&& view, auto res) { return res == std::ranges::begin(view) + std::ranges::size(view); },
        "adjacent_find, non-const callable");

#if TEST_DPCPP_BACKEND_PRESENT
    run_algo_hetero_policies<read_archetype_dc, 5>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::adjacent_find(std::forward<decltype(policy)>(policy), view, read_binary_pred_mut{});
        },
        [](auto&& view, auto res) { return res == std::ranges::begin(view) + std::ranges::size(view); },
        "adjacent_find, non-const callable");
#endif // TEST_DPCPP_BACKEND_PRESENT

    run_algo_host_policies<read_archetype>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::is_sorted(std::forward<decltype(policy)>(policy), view, read_comp_mut{});
        },
        [](auto&&, bool res) { return res; }, "is_sorted, non-const comparator");

#if TEST_DPCPP_BACKEND_PRESENT
    run_algo_hetero_policies<read_archetype_dc, 6>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::is_sorted(std::forward<decltype(policy)>(policy), view, read_comp_mut{});
        },
        [](auto&&, bool res) { return res; }, "is_sorted, non-const comparator");
#endif // TEST_DPCPP_BACKEND_PRESENT

    // The whole range is sorted, so the scan stops at its end.
    run_algo_host_policies<read_archetype>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::is_sorted_until(std::forward<decltype(policy)>(policy), view, read_comp_mut{});
        },
        [](auto&& view, auto res) { return res == std::ranges::begin(view) + std::ranges::size(view); },
        "is_sorted_until, non-const comparator");

#if TEST_DPCPP_BACKEND_PRESENT
    run_algo_hetero_policies<read_archetype_dc, 33>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::is_sorted_until(std::forward<decltype(policy)>(policy), view, read_comp_mut{});
        },
        [](auto&& view, auto res) { return res == std::ranges::begin(view) + std::ranges::size(view); },
        "is_sorted_until, non-const comparator");
#endif // TEST_DPCPP_BACKEND_PRESENT

    run_algo_host_policies<read_archetype>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::min_element(std::forward<decltype(policy)>(policy), view, read_comp_mut{});
        },
        [](auto&& view, auto res) { return res == std::ranges::begin(view); }, "min_element, non-const comparator");

#if TEST_DPCPP_BACKEND_PRESENT
    run_algo_hetero_policies<read_archetype_dc, 7>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::min_element(std::forward<decltype(policy)>(policy), view, read_comp_mut{});
        },
        [](auto&& view, auto res) { return res == std::ranges::begin(view); }, "min_element, non-const comparator");
#endif // TEST_DPCPP_BACKEND_PRESENT

    run_algo_host_policies<read_archetype>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::max_element(std::forward<decltype(policy)>(policy), view, read_comp_mut{});
        },
        [](auto&& view, auto res) { return res == std::ranges::begin(view) + std::ranges::size(view) - 1; },
        "max_element, non-const comparator");

#if TEST_DPCPP_BACKEND_PRESENT
    run_algo_hetero_policies<read_archetype_dc, 8>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::max_element(std::forward<decltype(policy)>(policy), view, read_comp_mut{});
        },
        [](auto&& view, auto res) { return res == std::ranges::begin(view) + std::ranges::size(view) - 1; },
        "max_element, non-const comparator");
#endif // TEST_DPCPP_BACKEND_PRESENT

    run_algo_host_policies<read_archetype>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::minmax_element(std::forward<decltype(policy)>(policy), view, read_comp_mut{});
        },
        [](auto&& view, auto res) {
            return res.min == std::ranges::begin(view) &&
                   res.max == std::ranges::begin(view) + std::ranges::size(view) - 1;
        },
        "minmax_element, non-const comparator");

#if TEST_DPCPP_BACKEND_PRESENT
    run_algo_hetero_policies<read_archetype_dc, 9>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::minmax_element(std::forward<decltype(policy)>(policy), view, read_comp_mut{});
        },
        [](auto&& view, auto res) {
            return res.min == std::ranges::begin(view) &&
                   res.max == std::ranges::begin(view) + std::ranges::size(view) - 1;
        },
        "minmax_element, non-const comparator");
#endif // TEST_DPCPP_BACKEND_PRESENT

    //----------------------------------------------------------------------------------------------
    // The value based algorithms with a projection taking the element by non-const reference.
    //----------------------------------------------------------------------------------------------
    run_algo_host_policies<searchable_archetype>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::find(std::forward<decltype(policy)>(policy), view, search_value{searched},
                                    search_proj_mut{});
        },
        [](auto&& view, auto res) { return res == std::ranges::begin(view) + searched; },
        "find, non-const projection");

#if TEST_DPCPP_BACKEND_PRESENT
    run_algo_hetero_policies<searchable_archetype_dc, 10>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::find(std::forward<decltype(policy)>(policy), view, search_value{searched},
                                    search_proj_mut{});
        },
        [](auto&& view, auto res) { return res == std::ranges::begin(view) + searched; },
        "find, non-const projection");
#endif // TEST_DPCPP_BACKEND_PRESENT

    run_algo_host_policies<searchable_archetype>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::find_last(std::forward<decltype(policy)>(policy), view, search_value{searched},
                                         search_proj_mut{});
        },
        [](auto&& view, auto res) { return std::ranges::begin(res) == std::ranges::begin(view) + searched; },
        "find_last, non-const projection");

#if TEST_DPCPP_BACKEND_PRESENT
    run_algo_hetero_policies<searchable_archetype_dc, 11>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::find_last(std::forward<decltype(policy)>(policy), view, search_value{searched},
                                         search_proj_mut{});
        },
        [](auto&& view, auto res) { return std::ranges::begin(res) == std::ranges::begin(view) + searched; },
        "find_last, non-const projection");
#endif // TEST_DPCPP_BACKEND_PRESENT

    run_algo_host_policies<searchable_archetype>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::count(std::forward<decltype(policy)>(policy), view, search_value{searched},
                                     search_proj_mut{});
        },
        [](auto&&, auto res) { return res == 1; }, "count, non-const projection");

#if TEST_DPCPP_BACKEND_PRESENT
    run_algo_hetero_policies<searchable_archetype_dc, 12>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::count(std::forward<decltype(policy)>(policy), view, search_value{searched},
                                     search_proj_mut{});
        },
        [](auto&&, auto res) { return res == 1; }, "count, non-const projection");
#endif // TEST_DPCPP_BACKEND_PRESENT

    run_algo_host_policies<searchable_archetype>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::contains(std::forward<decltype(policy)>(policy), view, search_value{searched},
                                        search_proj_mut{});
        },
        [](auto&&, auto res) { return res; }, "contains, non-const projection");

#if TEST_DPCPP_BACKEND_PRESENT
    run_algo_hetero_policies<searchable_archetype_dc, 13>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::contains(std::forward<decltype(policy)>(policy), view, search_value{searched},
                                        search_proj_mut{});
        },
        [](auto&&, auto res) { return res; }, "contains, non-const projection");
#endif // TEST_DPCPP_BACKEND_PRESENT

    run_algo_host_policies<removable_archetype>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::remove(std::forward<decltype(policy)>(policy), view, search_value{searched},
                                      search_proj_mut{});
        },
        // remove() returns the tail holding the removed elements, and the value 3 occurs exactly once.
        [](auto&&, auto res) { return std::ranges::size(res) == 1; }, "remove, non-const projection");

#if TEST_DPCPP_BACKEND_PRESENT
    run_algo_hetero_policies<removable_archetype_dc, 14>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::remove(std::forward<decltype(policy)>(policy), view, search_value{searched},
                                      search_proj_mut{});
        },
        // remove() returns the tail holding the removed elements, and the value 3 occurs exactly once.
        [](auto&&, auto res) { return std::ranges::size(res) == 1; }, "remove, non-const projection");
#endif // TEST_DPCPP_BACKEND_PRESENT

    //----------------------------------------------------------------------------------------------
    // Two-range algorithms with a predicate taking both elements by non-const reference.
    //----------------------------------------------------------------------------------------------
    run_algo2_host_policies<lhs_archetype, rhs_archetype>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            return dpl_ranges::equal(std::forward<decltype(policy)>(policy), view1, view2, cross_pred_mut{});
        },
        [](auto&&, auto&&, bool res) { return res; }, "equal, non-const callable");

#if TEST_DPCPP_BACKEND_PRESENT
    run_algo2_hetero_policies<lhs_archetype_dc, rhs_archetype_dc, 15>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            return dpl_ranges::equal(std::forward<decltype(policy)>(policy), view1, view2, cross_pred_mut{});
        },
        [](auto&&, auto&&, bool res) { return res; }, "equal, non-const callable");
#endif // TEST_DPCPP_BACKEND_PRESENT

    run_algo2_host_policies<lhs_archetype, rhs_archetype>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            return dpl_ranges::mismatch(std::forward<decltype(policy)>(policy), view1, view2, cross_pred_mut{});
        },
        [](auto&& view1, auto&& view2, auto res) {
            return res.in1 == std::ranges::begin(view1) + std::ranges::size(view1) &&
                   res.in2 == std::ranges::begin(view2) + std::ranges::size(view2);
        },
        "mismatch, non-const callable");

#if TEST_DPCPP_BACKEND_PRESENT
    run_algo2_hetero_policies<lhs_archetype_dc, rhs_archetype_dc, 16>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            return dpl_ranges::mismatch(std::forward<decltype(policy)>(policy), view1, view2, cross_pred_mut{});
        },
        [](auto&& view1, auto&& view2, auto res) {
            return res.in1 == std::ranges::begin(view1) + std::ranges::size(view1) &&
                   res.in2 == std::ranges::begin(view2) + std::ranges::size(view2);
        },
        "mismatch, non-const callable");
#endif // TEST_DPCPP_BACKEND_PRESENT

    // The two ranges hold the very same sequence, so the second one occurs in the first one exactly
    // once, at its very beginning.
    run_algo2_host_policies<lhs_archetype, rhs_archetype>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            return dpl_ranges::search(std::forward<decltype(policy)>(policy), view1, view2, cross_pred_mut{});
        },
        [](auto&& view1, auto&& view2, auto res) {
            return std::ranges::begin(res) == std::ranges::begin(view1) &&
                   std::ranges::size(res) == std::ranges::size(view2);
        },
        "search, non-const callable");

#if TEST_DPCPP_BACKEND_PRESENT
    run_algo2_hetero_policies<lhs_archetype_dc, rhs_archetype_dc, 34>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            return dpl_ranges::search(std::forward<decltype(policy)>(policy), view1, view2, cross_pred_mut{});
        },
        [](auto&& view1, auto&& view2, auto res) {
            return std::ranges::begin(res) == std::ranges::begin(view1) &&
                   std::ranges::size(res) == std::ranges::size(view2);
        },
        "search, non-const callable");
#endif // TEST_DPCPP_BACKEND_PRESENT

    run_algo2_host_policies<lhs_archetype, rhs_archetype>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            return dpl_ranges::find_end(std::forward<decltype(policy)>(policy), view1, view2, cross_pred_mut{});
        },
        [](auto&& view1, auto&& view2, auto res) {
            return std::ranges::begin(res) == std::ranges::begin(view1) &&
                   std::ranges::size(res) == std::ranges::size(view2);
        },
        "find_end, non-const callable");

#if TEST_DPCPP_BACKEND_PRESENT
    run_algo2_hetero_policies<lhs_archetype_dc, rhs_archetype_dc, 35>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            return dpl_ranges::find_end(std::forward<decltype(policy)>(policy), view1, view2, cross_pred_mut{});
        },
        [](auto&& view1, auto&& view2, auto res) {
            return std::ranges::begin(res) == std::ranges::begin(view1) &&
                   std::ranges::size(res) == std::ranges::size(view2);
        },
        "find_end, non-const callable");
#endif // TEST_DPCPP_BACKEND_PRESENT

    // KSATODO: std::indirectly_comparable<_It1, _It2, _Pred> only requires the predicate to be
    // invocable as __pred(*__it1, *__it2), never the other way round. The SIMD brick swaps the two
    // arguments, so the vectorized host policies unseq and par_unseq do not compile:
    //  - unseq_backend_simd.h:827 - __simd_find_first_of builds __u_pred as
    //    __pred(__val, *__first) with __val taken from the second range and *__first from the first
    //    one; the branch is a plain if, so it is instantiated whatever the sizes of the ranges are.
    // Fixing this means keeping the argument order of the two ranges in both branches.
    auto find_first_of_algo = [](auto&& policy, auto&& view1, auto&& view2) {
        return dpl_ranges::find_first_of(std::forward<decltype(policy)>(policy), view1, view2, cross_pred_mut{});
    };
    auto find_first_of_checker = [](auto&& view1, auto&&, auto res) { return res == std::ranges::begin(view1); };

#if !_TEST_CPP20_RANGES_BROKEN_REQUIRES_FIND_FIRST_OF_HOST
    run_algo2_host_policies<lhs_archetype, rhs_archetype>(find_first_of_algo, find_first_of_checker,
                                                          "find_first_of, non-const callable");
#endif

    // KSATODO: the device path of find_first_of copies the element of the first range into a const
    // local, which std::indirectly_comparable neither asks for nor allows to require, so the call does
    // not compile:
    //  - unseq_backend_sycl.h:632,636 - first_match_pred::operator() writes
    //    const auto __elem = __acc[__shifted_idx]; and passes __elem to the predicate. A forwarding
    //    reference instead of the const copy fixes both the const-ness and the extra copy.
#if TEST_DPCPP_BACKEND_PRESENT
#if !_TEST_CPP20_RANGES_BROKEN_REQUIRES_FIND_FIRST_OF_HETERO
    run_algo2_hetero_policies<lhs_archetype_dc, rhs_archetype_dc, 36>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            return dpl_ranges::find_first_of(std::forward<decltype(policy)>(policy), view1, view2, cross_pred_mut{});
        },
        [](auto&& view1, auto&&, auto res) { return res == std::ranges::begin(view1); },
        "find_first_of, non-const callable");
#endif // !_TEST_CPP20_RANGES_BROKEN_REQUIRES_FIND_FIRST_OF_HETERO
#endif // TEST_DPCPP_BACKEND_PRESENT

    // includes needs a comparator accepting the two element types in all four combinations, see
    // cross_comp_mut. Both ranges hold the very same ascending sequence, so the second one is included
    // in the first one.
    run_algo2_host_policies<lhs_archetype, rhs_archetype>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            return dpl_ranges::includes(std::forward<decltype(policy)>(policy), view1, view2, cross_comp_mut{});
        },
        [](auto&&, auto&&, bool res) { return res; }, "includes, non-const comparator");

#if TEST_DPCPP_BACKEND_PRESENT
    run_algo2_hetero_policies<lhs_archetype_dc, rhs_archetype_dc, 44>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            return dpl_ranges::includes(std::forward<decltype(policy)>(policy), view1, view2, cross_comp_mut{});
        },
        [](auto&&, auto&&, bool res) { return res; }, "includes, non-const comparator");
#endif // TEST_DPCPP_BACKEND_PRESENT

    //----------------------------------------------------------------------------------------------
    // transform with a functor taking the input element by non-const reference.
    //----------------------------------------------------------------------------------------------
    run_algo2_host_policies<transform_in_archetype, transform_out_archetype>(
        [](auto&& policy, auto&& in_view, auto&& out_view) {
            return dpl_ranges::transform(std::forward<decltype(policy)>(policy), in_view, out_view,
                                         transform_unary_op_mut{});
        },
        [](auto&&, auto&& out_view, auto) { return std::ranges::begin(out_view)[7].val == 14; },
        "transform, non-const callable");

#if TEST_DPCPP_BACKEND_PRESENT
    run_algo2_hetero_policies<transform_in_archetype_dc, transform_out_archetype_dc, 17>(
        [](auto&& policy, auto&& in_view, auto&& out_view) {
            return dpl_ranges::transform(std::forward<decltype(policy)>(policy), in_view, out_view,
                                         transform_unary_op_mut{});
        },
        [](auto&&, auto&& out_view, auto) { return std::ranges::begin(out_view)[7].val == 14; },
        "transform, non-const callable");
#endif // TEST_DPCPP_BACKEND_PRESENT

    //----------------------------------------------------------------------------------------------
    // The permuting and the sorting algorithms.
    //----------------------------------------------------------------------------------------------
    // Every third element is removed; the returned subrange is the tail holding the removed elements.
    run_algo_host_policies<permutable_archetype>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::remove_if(std::forward<decltype(policy)>(policy), view, permutable_pred_mut{});
        },
        [](auto&& view, auto res) { return std::ranges::size(res) == (std::ranges::size(view) + 2) / 3; },
        "remove_if, non-const callable");

#if TEST_DPCPP_BACKEND_PRESENT
    run_algo_hetero_policies<permutable_archetype_dc, 18>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::remove_if(std::forward<decltype(policy)>(policy), view, permutable_pred_mut{});
        },
        [](auto&& view, auto res) { return std::ranges::size(res) == (std::ranges::size(view) + 2) / 3; },
        "remove_if, non-const callable");
#endif // TEST_DPCPP_BACKEND_PRESENT

    // All the elements are unique, so nothing is dropped.
    run_algo_host_policies<permutable_archetype>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::unique(std::forward<decltype(policy)>(policy), view, permutable_equiv_mut{});
        },
        [](auto&&, auto res) { return std::ranges::size(res) == 0; }, "unique, non-const callable");

#if TEST_DPCPP_BACKEND_PRESENT
    run_algo_hetero_policies<permutable_archetype_dc, 19>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::unique(std::forward<decltype(policy)>(policy), view, permutable_equiv_mut{});
        },
        [](auto&&, auto res) { return std::ranges::size(res) == 0; }, "unique, non-const callable");
#endif // TEST_DPCPP_BACKEND_PRESENT

    // partition returns the tail of the elements which do not satisfy the predicate.
    run_algo_host_policies<permutable_archetype>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::partition(std::forward<decltype(policy)>(policy), view, permutable_pred_mut{});
        },
        [](auto&& view, auto res) {
            return std::ranges::size(res) == std::ranges::size(view) - (std::ranges::size(view) + 2) / 3;
        },
        "partition, non-const callable");

    // KSATODO: the device path of partition applies the predicate to a const lvalue, which
    // std::indirect_unary_predicate over a permutable iterator does not ask for, so it does not
    // compile:
    //  - unseq_backend_sycl.h:122 - walk_n::operator() is const and calls __f(__rngs[__idx]...) on
    //    the const range members of single_match_pred_by_idx.
#if TEST_DPCPP_BACKEND_PRESENT
#if !_TEST_CPP20_RANGES_BROKEN_REQUIRES_PARTITION_HETERO
    run_algo_hetero_policies<permutable_archetype_dc, 37>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::partition(std::forward<decltype(policy)>(policy), view, permutable_pred_mut{});
        },
        [](auto&& view, auto res) {
            return std::ranges::size(res) == std::ranges::size(view) - (std::ranges::size(view) + 2) / 3;
        },
        "partition, non-const callable");
#endif // !_TEST_CPP20_RANGES_BROKEN_REQUIRES_PARTITION_HETERO
#endif // TEST_DPCPP_BACKEND_PRESENT

    // The very same comparator as the one the sorting algorithms below are called with.
    run_algo_host_policies<permutable_archetype>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::is_sorted(std::forward<decltype(policy)>(policy), view, permutable_comp_mut{});
        },
        [](auto&&, bool res) { return res; }, "is_sorted of a permutable range, non-const comparator");

#if TEST_DPCPP_BACKEND_PRESENT
    run_algo_hetero_policies<permutable_archetype_dc, 38>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::is_sorted(std::forward<decltype(policy)>(policy), view, permutable_comp_mut{});
        },
        [](auto&&, bool res) { return res; }, "is_sorted of a permutable range, non-const comparator");
#endif // TEST_DPCPP_BACKEND_PRESENT

    // KSATODO: std::sortable<_It, _Comp> only requires the comparator to be invocable with
    // iter_reference_t<_It>, which is a non-const lvalue for archetype_view, so a comparator taking
    // its arguments by non-const reference is enough. The parallel host merge sort compares against a
    // const lvalue instead, so par and par_unseq do not compile:
    //  - parallel_backend_tbb.h:1037 - std::lower_bound(..., _M_comp) passes the const lvalue _Val
    //    of the merge split point to the comparator;
    //  - utils.h:203 - __binary_op::operator() forwards that const lvalue into std::invoke.
    // seq and unseq keep the element non-const all the way down.
    auto sort_algo = [](auto&& policy, auto&& view) {
        return dpl_ranges::sort(std::forward<decltype(policy)>(policy), view, permutable_comp_mut{});
    };
    auto sorted_checker = [](auto&& view, auto) {
        return std::ranges::begin(view)[0].val == 0 &&
               std::ranges::begin(view)[std::ranges::size(view) - 1].val == (int)std::ranges::size(view) - 1;
    };

#if !_TEST_CPP20_RANGES_BROKEN_REQUIRES_SORT_HOST
    run_algo_host_policies<permutable_archetype>(sort_algo, sorted_checker, "sort, non-const comparator");
#endif

#if TEST_DPCPP_BACKEND_PRESENT
    run_algo_hetero_policies<permutable_archetype_dc, 20>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::sort(std::forward<decltype(policy)>(policy), view, permutable_comp_mut{});
        },
        [](auto&& view, auto) {
            return std::ranges::begin(view)[0].val == 0 &&
                   std::ranges::begin(view)[std::ranges::size(view) - 1].val == (int)std::ranges::size(view) - 1;
        },
        "sort, non-const comparator");
#endif // TEST_DPCPP_BACKEND_PRESENT

    // KSATODO: stable_sort shares the merge sort of the parallel host policies with sort, so it is
    // broken for par and par_unseq in exactly the same way, see the note above.
    auto stable_sort_algo = [](auto&& policy, auto&& view) {
        return dpl_ranges::stable_sort(std::forward<decltype(policy)>(policy), view, permutable_comp_mut{});
    };

#if !_TEST_CPP20_RANGES_BROKEN_REQUIRES_STABLE_SORT_HOST
    run_algo_host_policies<permutable_archetype>(stable_sort_algo, sorted_checker, "stable_sort, non-const comparator");
#endif

#if TEST_DPCPP_BACKEND_PRESENT
    run_algo_hetero_policies<permutable_archetype_dc, 21>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::stable_sort(std::forward<decltype(policy)>(policy), view, permutable_comp_mut{});
        },
        [](auto&& view, auto) {
            return std::ranges::begin(view)[0].val == 0 &&
                   std::ranges::begin(view)[std::ranges::size(view) - 1].val == (int)std::ranges::size(view) - 1;
        },
        "stable_sort, non-const comparator");
#endif // TEST_DPCPP_BACKEND_PRESENT

    //----------------------------------------------------------------------------------------------
    // merge and min / max / minmax.
    //----------------------------------------------------------------------------------------------
    // Both inputs hold the very same sorted sequence 0, 1, 2, ...
    run_algo2_host_policies<merge_in_archetype, merge_in_archetype>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            using __out_elem = typename std::ranges::range_value_t<std::remove_cvref_t<decltype(view1)>>::out_type;
            archetype_storage<__out_elem, std::allocator<__out_elem>> out_storage(
                std::allocator<__out_elem>{}, 2 * archetype_test_size, [](std::size_t) { return 0; });
            auto out_view = out_storage.view();
            auto res =
                dpl_ranges::merge(std::forward<decltype(policy)>(policy), view1, view2, out_view, merge_comp_mut{});
            return std::ranges::begin(out_view)[0].val == 0 && std::ranges::begin(out_view)[1].val == 0 &&
                   std::ranges::begin(out_view)[2].val == 1 && res.out == std::ranges::end(out_view);
        },
        [](auto&&, auto&&, auto res) { return res; }, "merge, non-const comparator");

#if TEST_DPCPP_BACKEND_PRESENT
    run_algo2_hetero_policies<merge_in_archetype_dc, merge_in_archetype_dc, 22>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            using __out_elem = typename std::ranges::range_value_t<std::remove_cvref_t<decltype(view1)>>::out_type;
            // The output range is passed to a kernel just like the inputs, so it has to live in USM.
            sycl::usm_allocator<__out_elem, sycl::usm::alloc::shared> __out_alloc{policy.queue()};
            archetype_storage<__out_elem, decltype(__out_alloc)> out_storage(
                __out_alloc, 2 * archetype_test_size, [](std::size_t) { return 0; });
            auto out_view = out_storage.view();
            auto res =
                dpl_ranges::merge(std::forward<decltype(policy)>(policy), view1, view2, out_view, merge_comp_mut{});
            return std::ranges::begin(out_view)[0].val == 0 && std::ranges::begin(out_view)[1].val == 0 &&
                   std::ranges::begin(out_view)[2].val == 1 && res.out == std::ranges::end(out_view);
        },
        [](auto&&, auto&&, auto res) { return res; }, "merge, non-const comparator");
#endif // TEST_DPCPP_BACKEND_PRESENT

    run_algo_host_policies<storable_archetype>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::min(std::forward<decltype(policy)>(policy), view, storable_comp_mut{});
        },
        [](auto&&, auto res) { return res.val == 0; }, "min, non-const comparator");

#if TEST_DPCPP_BACKEND_PRESENT
    run_algo_hetero_policies<storable_archetype_dc, 23>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::min(std::forward<decltype(policy)>(policy), view, storable_comp_mut{});
        },
        [](auto&&, auto res) { return res.val == 0; }, "min, non-const comparator");
#endif // TEST_DPCPP_BACKEND_PRESENT

    run_algo_host_policies<storable_archetype>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::max(std::forward<decltype(policy)>(policy), view, storable_comp_mut{});
        },
        [](auto&&, auto res) { return res.val == (int)archetype_test_size - 1; }, "max, non-const comparator");

#if TEST_DPCPP_BACKEND_PRESENT
    run_algo_hetero_policies<storable_archetype_dc, 24>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::max(std::forward<decltype(policy)>(policy), view, storable_comp_mut{});
        },
        [](auto&&, auto res) { return res.val == (int)archetype_test_size - 1; }, "max, non-const comparator");
#endif // TEST_DPCPP_BACKEND_PRESENT

    run_algo_host_policies<storable_archetype>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::minmax(std::forward<decltype(policy)>(policy), view, storable_comp_mut{});
        },
        [](auto&&, auto&& res) { return res.min.val == 0 && res.max.val == (int)archetype_test_size - 1; },
        "minmax, non-const comparator");

#if TEST_DPCPP_BACKEND_PRESENT
    run_algo_hetero_policies<storable_archetype_dc, 25>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::minmax(std::forward<decltype(policy)>(policy), view, storable_comp_mut{});
        },
        [](auto&&, auto&& res) { return res.min.val == 0 && res.max.val == (int)archetype_test_size - 1; },
        "minmax, non-const comparator");
#endif // TEST_DPCPP_BACKEND_PRESENT

    //----------------------------------------------------------------------------------------------
    // The remaining algorithms constrained by std::sortable, i.e. by the very same comparator
    // requirement as sort: partial_sort, nth_element and inplace_merge.
    //----------------------------------------------------------------------------------------------
    // KSATODO: partial_sort shares the parallel merge sort with sort, so the parallel host policies
    // hand a const lvalue to the comparator here as well and par / par_unseq do not compile:
    //  - parallel_backend_tbb.h:1023,1026,1034 - __merge_func::split_merging passes *(_M_x_beg + __ym)
    //    to std::upper_bound / std::lower_bound, which compares against their const lvalue parameter;
    //  - utils.h:203 - __binary_op::operator() forwards that const lvalue into std::invoke.
    // seq and unseq keep the element non-const all the way down.
    // The range is ascending already, so the first ten elements are 0 ... 9 afterwards.
    auto partial_sort_algo = [](auto&& policy, auto&& view) {
        return dpl_ranges::partial_sort(std::forward<decltype(policy)>(policy), view, std::ranges::begin(view) + 10,
                                        permutable_comp_mut{});
    };
    auto partial_sort_checker = [](auto&& view, auto) {
        return std::ranges::begin(view)[0].val == 0 && std::ranges::begin(view)[9].val == 9;
    };

#if !_TEST_CPP20_RANGES_BROKEN_REQUIRES_PARTIAL_SORT_HOST
    run_algo_host_policies<permutable_archetype>(partial_sort_algo, partial_sort_checker,
                                                 "partial_sort, non-const comparator");
#endif

#if TEST_DPCPP_BACKEND_PRESENT
    run_algo_hetero_policies<permutable_archetype_dc, 39>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::partial_sort(std::forward<decltype(policy)>(policy), view,
                                            std::ranges::begin(view) + 10, permutable_comp_mut{});
        },
        [](auto&& view, auto) { return std::ranges::begin(view)[0].val == 0 && std::ranges::begin(view)[9].val == 9; },
        "partial_sort, non-const comparator");
#endif // TEST_DPCPP_BACKEND_PRESENT

    // KSATODO: std::sortable only requires the comparator to be invocable with the non-const
    // reference of the element, but the parallel path of nth_element compares against a const lvalue,
    // so par and par_unseq do not compile:
    //  - algorithm_impl.h:2841 - the partition predicate of the quickselect loop takes const _Tp& and
    //    passes it into std::invoke(__comp, __x, *__first).
    // Taking the element by reference in that lambda is enough to fix it; seq and unseq are fine.
    auto nth_element_algo = [](auto&& policy, auto&& view) {
        return dpl_ranges::nth_element(std::forward<decltype(policy)>(policy), view, std::ranges::begin(view) + 10,
                                       permutable_comp_mut{});
    };
    auto nth_element_checker = [](auto&& view, auto) { return std::ranges::begin(view)[10].val == 10; };

#if !_TEST_CPP20_RANGES_BROKEN_REQUIRES_NTH_ELEMENT_HOST
    run_algo_host_policies<permutable_archetype>(nth_element_algo, nth_element_checker,
                                                 "nth_element, non-const comparator");
#endif

#if TEST_DPCPP_BACKEND_PRESENT
    run_algo_hetero_policies<permutable_archetype_dc, 40>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::nth_element(std::forward<decltype(policy)>(policy), view, std::ranges::begin(view) + 10,
                                           permutable_comp_mut{});
        },
        [](auto&& view, auto) { return std::ranges::begin(view)[10].val == 10; }, "nth_element, non-const comparator");
#endif // TEST_DPCPP_BACKEND_PRESENT

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
#if !_TEST_CPP20_RANGES_BROKEN_REQUIRES_INPLACE_MERGE_HOST
    run_algo_host_policies<permutable_archetype>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::inplace_merge(std::forward<decltype(policy)>(policy), view,
                                             std::ranges::begin(view) + std::ranges::size(view) / 2,
                                             permutable_comp_mut{});
        },
        sorted_checker, "inplace_merge, non-const comparator");
#endif

    // KSATODO: the device path of inplace_merge compares two const lvalues, which std::sortable does
    // not ask for, so the call does not compile:
    //  - parallel_backend_sycl_merge.h:128-133 - the lambda of __find_start_point captures __rng1 and
    //    __rng2 and subscripts them as const, and both results go into the comparator.
#if TEST_DPCPP_BACKEND_PRESENT
#if !_TEST_CPP20_RANGES_BROKEN_REQUIRES_INPLACE_MERGE_HETERO
    run_algo_hetero_policies<permutable_archetype_dc, 41>(
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
#endif // !_TEST_CPP20_RANGES_BROKEN_REQUIRES_INPLACE_MERGE_HETERO
#endif // TEST_DPCPP_BACKEND_PRESENT

    //----------------------------------------------------------------------------------------------
    // The set operations, whose comparator is constrained exactly like the one of merge. They are
    // guarded by the very same macros as in std_ranges_algo_archetypes_merge.pass.cpp: the
    // implementation constructs the output element instead of assigning to it, which std::mergeable
    // never asks for, and that breaks the call before the comparator is ever reached.
    //----------------------------------------------------------------------------------------------
#if !_TEST_CPP20_RANGES_BROKEN_REQUIRES_SET_UNION_HOST
    run_algo2_host_policies<merge_in_archetype, merge_in_archetype>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            archetype_storage<merge_out_archetype, std::allocator<merge_out_archetype>> out_storage(
                std::allocator<merge_out_archetype>{}, 2 * archetype_test_size, [](std::size_t) { return 0; });
            auto out_view = out_storage.view();
            auto res = dpl_ranges::set_union(std::forward<decltype(policy)>(policy), view1, view2, out_view,
                                             merge_comp_mut{});
            // The two inputs hold the very same sequence, so the union is that sequence itself.
            return std::ranges::begin(out_view)[7].val == 7 &&
                   (std::size_t)(res.out - std::ranges::begin(out_view)) == archetype_test_size;
        },
        [](auto&&, auto&&, auto res) { return res; }, "set_union, non-const comparator");
#endif

#if !_TEST_CPP20_RANGES_BROKEN_REQUIRES_SET_DIFFERENCE_HOST
    run_algo2_host_policies<merge_in_archetype, merge_in_archetype>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            archetype_storage<merge_out_archetype, std::allocator<merge_out_archetype>> out_storage(
                std::allocator<merge_out_archetype>{}, 2 * archetype_test_size, [](std::size_t) { return 0; });
            auto out_view = out_storage.view();
            auto res = dpl_ranges::set_difference(std::forward<decltype(policy)>(policy), view1, view2, out_view,
                                                 merge_comp_mut{});
            // The two inputs are equal, so the difference is empty.
            return res.out == std::ranges::begin(out_view);
        },
        [](auto&&, auto&&, auto res) { return res; }, "set_difference, non-const comparator");
#endif

#if TEST_DPCPP_BACKEND_PRESENT
#if !_TEST_CPP20_RANGES_BROKEN_REQUIRES_SET_UNION_HETERO
    run_algo2_hetero_policies<merge_in_archetype_dc, merge_in_archetype_dc, 42>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            using __out_elem = typename std::ranges::range_value_t<std::remove_cvref_t<decltype(view1)>>::out_type;
            sycl::usm_allocator<__out_elem, sycl::usm::alloc::shared> __out_alloc{policy.queue()};
            archetype_storage<__out_elem, decltype(__out_alloc)> out_storage(__out_alloc, 2 * archetype_test_size,
                                                                            [](std::size_t) { return 0; });
            auto out_view = out_storage.view();
            auto res = dpl_ranges::set_union(std::forward<decltype(policy)>(policy), view1, view2, out_view,
                                             merge_comp_mut{});
            return std::ranges::begin(out_view)[7].val == 7 &&
                   (std::size_t)(res.out - std::ranges::begin(out_view)) == archetype_test_size;
        },
        [](auto&&, auto&&, auto res) { return res; }, "set_union, non-const comparator");
#endif // !_TEST_CPP20_RANGES_BROKEN_REQUIRES_SET_UNION_HETERO

#if !_TEST_CPP20_RANGES_BROKEN_REQUIRES_SET_DIFFERENCE_HETERO
    run_algo2_hetero_policies<merge_in_archetype_dc, merge_in_archetype_dc, 43>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            using __out_elem = typename std::ranges::range_value_t<std::remove_cvref_t<decltype(view1)>>::out_type;
            sycl::usm_allocator<__out_elem, sycl::usm::alloc::shared> __out_alloc{policy.queue()};
            archetype_storage<__out_elem, decltype(__out_alloc)> out_storage(__out_alloc, 2 * archetype_test_size,
                                                                            [](std::size_t) { return 0; });
            auto out_view = out_storage.view();
            auto res = dpl_ranges::set_difference(std::forward<decltype(policy)>(policy), view1, view2, out_view,
                                                 merge_comp_mut{});
            return res.out == std::ranges::begin(out_view);
        },
        [](auto&&, auto&&, auto res) { return res; }, "set_difference, non-const comparator");
#endif // !_TEST_CPP20_RANGES_BROKEN_REQUIRES_SET_DIFFERENCE_HETERO
#endif // TEST_DPCPP_BACKEND_PRESENT

#endif //_ENABLE_STD_RANGES_TESTING

    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
