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

    // read_archetype is neither copyable, movable, default constructible nor comparable; the only
    // operations available are the ones the callables of the algorithm provide.
    run_algo_host_policies<read_archetype>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::for_each(std::forward<decltype(policy)>(policy), view, read_unary_fun{});
        },
        [](auto&& view, auto res) { return res == std::ranges::begin(view) + std::ranges::size(view); }, "for_each");

#if TEST_DPCPP_BACKEND_PRESENT
    run_algo_hetero_policies<read_archetype_dc, 15>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::for_each(std::forward<decltype(policy)>(policy), view, read_unary_fun{});
        },
        [](auto&& view, auto res) { return res == std::ranges::begin(view) + std::ranges::size(view); }, "for_each");
#endif // TEST_DPCPP_BACKEND_PRESENT

    run_algo_host_policies<read_archetype>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::find_if(std::forward<decltype(policy)>(policy), view, read_unary_pred{});
        },
        [](auto&& view, auto res) { return res == std::ranges::begin(view); }, "find_if");

#if TEST_DPCPP_BACKEND_PRESENT
    run_algo_hetero_policies<read_archetype_dc, 0>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::find_if(std::forward<decltype(policy)>(policy), view, read_unary_pred{});
        },
        [](auto&& view, auto res) { return res == std::ranges::begin(view); }, "find_if");
#endif // TEST_DPCPP_BACKEND_PRESENT

    run_algo_host_policies<read_archetype>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::find_if_not(std::forward<decltype(policy)>(policy), view, read_unary_pred{});
        },
        [](auto&& view, auto res) { return res == std::ranges::begin(view) + 1; }, "find_if_not");

#if TEST_DPCPP_BACKEND_PRESENT
    run_algo_hetero_policies<read_archetype_dc, 1>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::find_if_not(std::forward<decltype(policy)>(policy), view, read_unary_pred{});
        },
        [](auto&& view, auto res) { return res == std::ranges::begin(view) + 1; }, "find_if_not");
#endif // TEST_DPCPP_BACKEND_PRESENT

    // The last element whose value is divisible by three, and the last one whose value is not.
    run_algo_host_policies<read_archetype>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::find_last_if(std::forward<decltype(policy)>(policy), view, read_unary_pred{});
        },
        [](auto&& view, auto res) {
            auto __n = (int)std::ranges::size(view);
            return std::ranges::begin(res) == std::ranges::begin(view) + (__n - 1) / 3 * 3;
        },
        "find_last_if");

#if TEST_DPCPP_BACKEND_PRESENT
    run_algo_hetero_policies<read_archetype_dc, 16>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::find_last_if(std::forward<decltype(policy)>(policy), view, read_unary_pred{});
        },
        [](auto&& view, auto res) {
            auto __n = (int)std::ranges::size(view);
            return std::ranges::begin(res) == std::ranges::begin(view) + (__n - 1) / 3 * 3;
        },
        "find_last_if");
#endif // TEST_DPCPP_BACKEND_PRESENT

    run_algo_host_policies<read_archetype>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::find_last_if_not(std::forward<decltype(policy)>(policy), view, read_unary_pred{});
        },
        [](auto&& view, auto res) {
            auto __n = (int)std::ranges::size(view);
            return std::ranges::begin(res) == std::ranges::begin(view) + ((__n - 1) % 3 == 0 ? __n - 2 : __n - 1);
        },
        "find_last_if_not");

#if TEST_DPCPP_BACKEND_PRESENT
    run_algo_hetero_policies<read_archetype_dc, 17>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::find_last_if_not(std::forward<decltype(policy)>(policy), view, read_unary_pred{});
        },
        [](auto&& view, auto res) {
            auto __n = (int)std::ranges::size(view);
            return std::ranges::begin(res) == std::ranges::begin(view) + ((__n - 1) % 3 == 0 ? __n - 2 : __n - 1);
        },
        "find_last_if_not");
#endif // TEST_DPCPP_BACKEND_PRESENT

    run_algo_host_policies<read_archetype>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::any_of(std::forward<decltype(policy)>(policy), view, read_unary_pred{});
        },
        [](auto&&, bool res) { return res; }, "any_of");

#if TEST_DPCPP_BACKEND_PRESENT
    run_algo_hetero_policies<read_archetype_dc, 2>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::any_of(std::forward<decltype(policy)>(policy), view, read_unary_pred{});
        },
        [](auto&&, bool res) { return res; }, "any_of");
#endif // TEST_DPCPP_BACKEND_PRESENT

    run_algo_host_policies<read_archetype>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::all_of(std::forward<decltype(policy)>(policy), view, read_unary_pred{});
        },
        [](auto&&, bool res) { return !res; }, "all_of");

#if TEST_DPCPP_BACKEND_PRESENT
    run_algo_hetero_policies<read_archetype_dc, 3>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::all_of(std::forward<decltype(policy)>(policy), view, read_unary_pred{});
        },
        [](auto&&, bool res) { return !res; }, "all_of");
#endif // TEST_DPCPP_BACKEND_PRESENT

    run_algo_host_policies<read_archetype>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::none_of(std::forward<decltype(policy)>(policy), view, read_unary_pred{});
        },
        [](auto&&, bool res) { return !res; }, "none_of");

#if TEST_DPCPP_BACKEND_PRESENT
    run_algo_hetero_policies<read_archetype_dc, 18>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::none_of(std::forward<decltype(policy)>(policy), view, read_unary_pred{});
        },
        [](auto&&, bool res) { return !res; }, "none_of");
#endif // TEST_DPCPP_BACKEND_PRESENT

    // The predicate holds for 0, fails for 1 and holds again for 3, so the range is not partitioned.
    run_algo_host_policies<read_archetype>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::is_partitioned(std::forward<decltype(policy)>(policy), view, read_unary_pred{});
        },
        [](auto&&, bool res) { return !res; }, "is_partitioned");

#if TEST_DPCPP_BACKEND_PRESENT
    run_algo_hetero_policies<read_archetype_dc, 19>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::is_partitioned(std::forward<decltype(policy)>(policy), view, read_unary_pred{});
        },
        [](auto&&, bool res) { return !res; }, "is_partitioned");
#endif // TEST_DPCPP_BACKEND_PRESENT

    run_algo_host_policies<read_archetype>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::count_if(std::forward<decltype(policy)>(policy), view, read_unary_pred{});
        },
        [](auto&& view, auto res) { return res == (std::ranges::range_difference_t<decltype(view)>)
                                                      ((std::ranges::size(view) + 2) / 3); }, "count_if");

#if TEST_DPCPP_BACKEND_PRESENT
    run_algo_hetero_policies<read_archetype_dc, 4>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::count_if(std::forward<decltype(policy)>(policy), view, read_unary_pred{});
        },
        [](auto&& view, auto res) { return res == (std::ranges::range_difference_t<decltype(view)>)
                                                      ((std::ranges::size(view) + 2) / 3); }, "count_if");
#endif // TEST_DPCPP_BACKEND_PRESENT

    // The projection returns an unrelated prvalue type, so the predicate can only ever be applied to
    // the projected value.
    run_algo_host_policies<read_archetype>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::find_if(std::forward<decltype(policy)>(policy), view, read_proj_pred{}, read_proj{});
        },
        [](auto&& view, auto res) { return res == std::ranges::begin(view); }, "find_if with proj");

#if TEST_DPCPP_BACKEND_PRESENT
    run_algo_hetero_policies<read_archetype_dc, 20>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::find_if(std::forward<decltype(policy)>(policy), view, read_proj_pred{}, read_proj{});
        },
        [](auto&& view, auto res) { return res == std::ranges::begin(view); }, "find_if with proj");
#endif // TEST_DPCPP_BACKEND_PRESENT

    run_algo_host_policies<read_archetype>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::count_if(std::forward<decltype(policy)>(policy), view, read_proj_pred{}, read_proj{});
        },
        [](auto&& view, auto res) { return res == (std::ranges::range_difference_t<decltype(view)>)
                                                      ((std::ranges::size(view) + 2) / 3); }, "count_if with proj");

#if TEST_DPCPP_BACKEND_PRESENT
    run_algo_hetero_policies<read_archetype_dc, 5>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::count_if(std::forward<decltype(policy)>(policy), view, read_proj_pred{}, read_proj{});
        },
        [](auto&& view, auto res) { return res == (std::ranges::range_difference_t<decltype(view)>)
                                                      ((std::ranges::size(view) + 2) / 3); }, "count_if with proj");
#endif // TEST_DPCPP_BACKEND_PRESENT

    // min_element/max_element/minmax_element only require std::indirect_strict_weak_order on the
    // projected iterator, so the element type stays non-copyable and non-default-constructible: both
    // backends carry an index and dereference the iterator for the comparison instead of storing the
    // element by value.
    run_algo_host_policies<read_archetype>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::min_element(std::forward<decltype(policy)>(policy), view, read_comp{});
        },
        [](auto&& view, auto res) { return res == std::ranges::begin(view); }, "min_element");

#if TEST_DPCPP_BACKEND_PRESENT
    run_algo_hetero_policies<read_archetype_dc, 12>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::min_element(std::forward<decltype(policy)>(policy), view, read_comp{});
        },
        [](auto&& view, auto res) { return res == std::ranges::begin(view); }, "min_element");
#endif // TEST_DPCPP_BACKEND_PRESENT

    run_algo_host_policies<read_archetype>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::max_element(std::forward<decltype(policy)>(policy), view, read_comp{});
        },
        [](auto&& view, auto res) { return res == std::ranges::begin(view) + std::ranges::size(view) - 1; },
        "max_element");

#if TEST_DPCPP_BACKEND_PRESENT
    run_algo_hetero_policies<read_archetype_dc, 13>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::max_element(std::forward<decltype(policy)>(policy), view, read_comp{});
        },
        [](auto&& view, auto res) { return res == std::ranges::begin(view) + std::ranges::size(view) - 1; },
        "max_element");
#endif // TEST_DPCPP_BACKEND_PRESENT

    run_algo_host_policies<read_archetype>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::minmax_element(std::forward<decltype(policy)>(policy), view, read_comp{});
        },
        [](auto&& view, auto res) {
            return res.min == std::ranges::begin(view) &&
                   res.max == std::ranges::begin(view) + std::ranges::size(view) - 1;
        },
        "minmax_element");


#if TEST_DPCPP_BACKEND_PRESENT
    run_algo_hetero_policies<read_archetype_dc, 14>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::minmax_element(std::forward<decltype(policy)>(policy), view, read_comp{});
        },
        [](auto&& view, auto res) {
            return res.min == std::ranges::begin(view) &&
                   res.max == std::ranges::begin(view) + std::ranges::size(view) - 1;
        },
        "minmax_element");
#endif // TEST_DPCPP_BACKEND_PRESENT

    run_algo_host_policies<read_archetype>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::is_sorted(std::forward<decltype(policy)>(policy), view, read_comp{});
        },
        [](auto&&, bool res) { return res; }, "is_sorted");

#if TEST_DPCPP_BACKEND_PRESENT
    run_algo_hetero_policies<read_archetype_dc, 6>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::is_sorted(std::forward<decltype(policy)>(policy), view, read_comp{});
        },
        [](auto&&, bool res) { return res; }, "is_sorted");
#endif // TEST_DPCPP_BACKEND_PRESENT

    // The whole range is sorted, so the scan stops at its end.
    run_algo_host_policies<read_archetype>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::is_sorted_until(std::forward<decltype(policy)>(policy), view, read_comp{});
        },
        [](auto&& view, auto res) { return res == std::ranges::begin(view) + std::ranges::size(view); },
        "is_sorted_until");

#if TEST_DPCPP_BACKEND_PRESENT
    run_algo_hetero_policies<read_archetype_dc, 21>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::is_sorted_until(std::forward<decltype(policy)>(policy), view, read_comp{});
        },
        [](auto&& view, auto res) { return res == std::ranges::begin(view) + std::ranges::size(view); },
        "is_sorted_until");
#endif // TEST_DPCPP_BACKEND_PRESENT

    run_algo_host_policies<read_archetype>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::adjacent_find(std::forward<decltype(policy)>(policy), view, read_binary_pred{});
        },
        [](auto&& view, auto res) { return res == std::ranges::begin(view) + std::ranges::size(view); },
        "adjacent_find");

#if TEST_DPCPP_BACKEND_PRESENT
    run_algo_hetero_policies<read_archetype_dc, 7>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::adjacent_find(std::forward<decltype(policy)>(policy), view, read_binary_pred{});
        },
        [](auto&& view, auto res) { return res == std::ranges::begin(view) + std::ranges::size(view); },
        "adjacent_find");
#endif // TEST_DPCPP_BACKEND_PRESENT

    // The search value type is unrelated to the element type.
    run_algo_host_policies<searchable_archetype>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::find(std::forward<decltype(policy)>(policy), view, search_value{7});
        },
        [](auto&& view, auto res) { return res == std::ranges::begin(view) + 7; }, "find");

#if TEST_DPCPP_BACKEND_PRESENT
    run_algo_hetero_policies<searchable_archetype_dc, 8>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::find(std::forward<decltype(policy)>(policy), view, search_value{7});
        },
        [](auto&& view, auto res) { return res == std::ranges::begin(view) + 7; }, "find");
#endif // TEST_DPCPP_BACKEND_PRESENT

    // find_last returns the tail of the range starting at the last occurrence of the value.
    run_algo_host_policies<searchable_archetype>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::find_last(std::forward<decltype(policy)>(policy), view, search_value{7});
        },
        [](auto&& view, auto res) {
            return std::ranges::begin(res) == std::ranges::begin(view) + 7 &&
                   std::ranges::size(res) == (std::ranges::range_difference_t<decltype(view)>)std::ranges::size(view) - 7;
        },
        "find_last");

#if TEST_DPCPP_BACKEND_PRESENT
    run_algo_hetero_policies<searchable_archetype_dc, 22>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::find_last(std::forward<decltype(policy)>(policy), view, search_value{7});
        },
        [](auto&& view, auto res) {
            return std::ranges::begin(res) == std::ranges::begin(view) + 7 &&
                   std::ranges::size(res) == (std::ranges::range_difference_t<decltype(view)>)std::ranges::size(view) - 7;
        },
        "find_last");
#endif // TEST_DPCPP_BACKEND_PRESENT

    run_algo_host_policies<searchable_archetype>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::count(std::forward<decltype(policy)>(policy), view, search_value{7});
        },
        [](auto&&, auto res) { return res == 1; }, "count");

#if TEST_DPCPP_BACKEND_PRESENT
    run_algo_hetero_policies<searchable_archetype_dc, 9>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::count(std::forward<decltype(policy)>(policy), view, search_value{7});
        },
        [](auto&&, auto res) { return res == 1; }, "count");
#endif // TEST_DPCPP_BACKEND_PRESENT

    // Two ranges of unrelated element types, compared only through the user predicate.
    run_algo2_host_policies<lhs_archetype, rhs_archetype>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            return dpl_ranges::equal(std::forward<decltype(policy)>(policy), view1, view2, cross_pred{});
        },
        [](auto&&, auto&&, bool res) { return res; }, "equal");

#if TEST_DPCPP_BACKEND_PRESENT
    run_algo2_hetero_policies<lhs_archetype_dc, rhs_archetype_dc, 10>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            return dpl_ranges::equal(std::forward<decltype(policy)>(policy), view1, view2, cross_pred{});
        },
        [](auto&&, auto&&, bool res) { return res; }, "equal");
#endif // TEST_DPCPP_BACKEND_PRESENT

    run_algo2_host_policies<lhs_archetype, rhs_archetype>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            return dpl_ranges::mismatch(std::forward<decltype(policy)>(policy), view1, view2, cross_pred{});
        },
        [](auto&& view1, auto&& view2, auto res) {
            return res.in1 == std::ranges::begin(view1) + std::ranges::size(view1) &&
                   res.in2 == std::ranges::begin(view2) + std::ranges::size(view2);
        },
        "mismatch");

#if TEST_DPCPP_BACKEND_PRESENT
    run_algo2_hetero_policies<lhs_archetype_dc, rhs_archetype_dc, 11>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            return dpl_ranges::mismatch(std::forward<decltype(policy)>(policy), view1, view2, cross_pred{});
        },
        [](auto&& view1, auto&& view2, auto res) {
            return res.in1 == std::ranges::begin(view1) + std::ranges::size(view1) &&
                   res.in2 == std::ranges::begin(view2) + std::ranges::size(view2);
        },
        "mismatch");
#endif // TEST_DPCPP_BACKEND_PRESENT

    // The two ranges hold the very same sequence, so the second one occurs in the first one exactly
    // once, at its very beginning.
    run_algo2_host_policies<lhs_archetype, rhs_archetype>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            return dpl_ranges::search(std::forward<decltype(policy)>(policy), view1, view2, cross_pred{});
        },
        [](auto&& view1, auto&& view2, auto res) {
            return std::ranges::begin(res) == std::ranges::begin(view1) &&
                   std::ranges::size(res) == std::ranges::size(view2);
        },
        "search");

#if TEST_DPCPP_BACKEND_PRESENT
    run_algo2_hetero_policies<lhs_archetype_dc, rhs_archetype_dc, 23>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            return dpl_ranges::search(std::forward<decltype(policy)>(policy), view1, view2, cross_pred{});
        },
        [](auto&& view1, auto&& view2, auto res) {
            return std::ranges::begin(res) == std::ranges::begin(view1) &&
                   std::ranges::size(res) == std::ranges::size(view2);
        },
        "search");
#endif // TEST_DPCPP_BACKEND_PRESENT

    run_algo2_host_policies<lhs_archetype, rhs_archetype>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            return dpl_ranges::find_end(std::forward<decltype(policy)>(policy), view1, view2, cross_pred{});
        },
        [](auto&& view1, auto&& view2, auto res) {
            return std::ranges::begin(res) == std::ranges::begin(view1) &&
                   std::ranges::size(res) == std::ranges::size(view2);
        },
        "find_end");

#if TEST_DPCPP_BACKEND_PRESENT
    run_algo2_hetero_policies<lhs_archetype_dc, rhs_archetype_dc, 24>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            return dpl_ranges::find_end(std::forward<decltype(policy)>(policy), view1, view2, cross_pred{});
        },
        [](auto&& view1, auto&& view2, auto res) {
            return std::ranges::begin(res) == std::ranges::begin(view1) &&
                   std::ranges::size(res) == std::ranges::size(view2);
        },
        "find_end");
#endif // TEST_DPCPP_BACKEND_PRESENT

    // KSATODO: std::indirectly_comparable<_It1, _It2, _Pred> only requires the predicate to be
    // invocable as __pred(*__it1, *__it2), never the other way round. The SIMD brick swaps the two
    // arguments, so the vectorized host policies unseq and par_unseq do not compile:
    //  - unseq_backend_simd.h:827 - __simd_find_first_of builds __u_pred as
    //    __pred(__val, *__first) with __val taken from the second range and *__first from the first
    //    one; the branch is a plain if, so it is instantiated whatever the sizes of the ranges are.
    // Fixing this means keeping the argument order of the two ranges in both branches.
    auto find_first_of_algo = [](auto&& policy, auto&& view1, auto&& view2) {
        return dpl_ranges::find_first_of(std::forward<decltype(policy)>(policy), view1, view2, cross_pred{});
    };
    auto find_first_of_checker = [](auto&& view1, auto&&, auto res) { return res == std::ranges::begin(view1); };

#if !_TEST_CPP20_RANGES_BROKEN_REQUIRES_FIND_FIRST_OF_HOST
    run_algo2_host_policies<lhs_archetype, rhs_archetype>(find_first_of_algo, find_first_of_checker, "find_first_of");
#endif

#if TEST_DPCPP_BACKEND_PRESENT
    run_algo2_hetero_policies<lhs_archetype_dc, rhs_archetype_dc, 25>(find_first_of_algo, find_first_of_checker,
                                                                      "find_first_of");
#endif // TEST_DPCPP_BACKEND_PRESENT

#endif //_ENABLE_STD_RANGES_TESTING

    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
