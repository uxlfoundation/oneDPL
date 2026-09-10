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
    // is_partitioned, count_if, min_element, max_element, minmax_element, is_sorted, is_sorted_until,
    // is_heap, is_heap_until and adjacent_find, first with const callables, then with callables taking
    // non-const references, and finally without a callable at all, i.e. with the default
    // std::ranges::less and std::ranges::equal_to.

    // read_archetype is neither copyable, movable, default constructible nor comparable; the only
    // operations available are the ones the callables of the algorithm provide.
    run_algo_all_policies<read_archetype, read_archetype_dc, 0>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::for_each(std::forward<decltype(policy)>(policy), view, read_unary_fun{});
        },
        [](auto&& view, auto res) { return res == std::ranges::begin(view) + std::ranges::size(view); }, "for_each");

    run_algo_all_policies<read_archetype, read_archetype_dc, 1>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::find_if(std::forward<decltype(policy)>(policy), view, read_unary_pred{});
        },
        [](auto&& view, auto res) { return res == std::ranges::begin(view); }, "find_if");

    run_algo_all_policies<read_archetype, read_archetype_dc, 2>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::find_if_not(std::forward<decltype(policy)>(policy), view, read_unary_pred{});
        },
        [](auto&& view, auto res) { return res == std::ranges::begin(view) + 1; }, "find_if_not");

    // The last element whose value is divisible by three, and the last one whose value is not.
    run_algo_all_policies<read_archetype, read_archetype_dc, 3>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::find_last_if(std::forward<decltype(policy)>(policy), view, read_unary_pred{});
        },
        [](auto&& view, auto res) {
            auto __n = (int)std::ranges::size(view);
            return std::ranges::begin(res) == std::ranges::begin(view) + (__n - 1) / 3 * 3;
        },
        "find_last_if");

    run_algo_all_policies<read_archetype, read_archetype_dc, 4>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::find_last_if_not(std::forward<decltype(policy)>(policy), view, read_unary_pred{});
        },
        [](auto&& view, auto res) {
            auto __n = (int)std::ranges::size(view);
            return std::ranges::begin(res) == std::ranges::begin(view) + ((__n - 1) % 3 == 0 ? __n - 2 : __n - 1);
        },
        "find_last_if_not");

    run_algo_all_policies<read_archetype, read_archetype_dc, 5>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::any_of(std::forward<decltype(policy)>(policy), view, read_unary_pred{});
        },
        [](auto&&, bool res) { return res; }, "any_of");

    run_algo_all_policies<read_archetype, read_archetype_dc, 6>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::all_of(std::forward<decltype(policy)>(policy), view, read_unary_pred{});
        },
        [](auto&&, bool res) { return !res; }, "all_of");

    run_algo_all_policies<read_archetype, read_archetype_dc, 7>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::none_of(std::forward<decltype(policy)>(policy), view, read_unary_pred{});
        },
        [](auto&&, bool res) { return !res; }, "none_of");

    // The predicate holds for 0, fails for 1 and holds again for 3, so the range is not partitioned.
    run_algo_all_policies<read_archetype, read_archetype_dc, 8>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::is_partitioned(std::forward<decltype(policy)>(policy), view, read_unary_pred{});
        },
        [](auto&&, bool res) { return !res; }, "is_partitioned");

    run_algo_all_policies<read_archetype, read_archetype_dc, 9>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::count_if(std::forward<decltype(policy)>(policy), view, read_unary_pred{});
        },
        [](auto&& view, auto res) { return res == (std::ranges::range_difference_t<decltype(view)>)
                                                      ((std::ranges::size(view) + 2) / 3); }, "count_if");

    // The projection returns an unrelated prvalue type, so the predicate can only ever be applied to
    // the projected value.
    run_algo_all_policies<read_archetype, read_archetype_dc, 10>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::find_if(std::forward<decltype(policy)>(policy), view, read_proj_pred{}, read_proj{});
        },
        [](auto&& view, auto res) { return res == std::ranges::begin(view); }, "find_if with proj");

    run_algo_all_policies<read_archetype, read_archetype_dc, 11>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::count_if(std::forward<decltype(policy)>(policy), view, read_proj_pred{}, read_proj{});
        },
        [](auto&& view, auto res) { return res == (std::ranges::range_difference_t<decltype(view)>)
                                                      ((std::ranges::size(view) + 2) / 3); }, "count_if with proj");

    // min_element/max_element/minmax_element only require std::indirect_strict_weak_order on the
    // projected iterator, so the element type stays non-copyable and non-default-constructible: both
    // backends carry an index and dereference the iterator for the comparison instead of storing the
    // element by value.
    run_algo_all_policies<read_archetype, read_archetype_dc, 12>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::min_element(std::forward<decltype(policy)>(policy), view, read_comp{});
        },
        [](auto&& view, auto res) { return res == std::ranges::begin(view); }, "min_element");

    run_algo_all_policies<read_archetype, read_archetype_dc, 13>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::max_element(std::forward<decltype(policy)>(policy), view, read_comp{});
        },
        [](auto&& view, auto res) { return res == std::ranges::begin(view) + std::ranges::size(view) - 1; },
        "max_element");

    run_algo_all_policies<read_archetype, read_archetype_dc, 14>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::minmax_element(std::forward<decltype(policy)>(policy), view, read_comp{});
        },
        [](auto&& view, auto res) {
            return res.min == std::ranges::begin(view) &&
                   res.max == std::ranges::begin(view) + std::ranges::size(view) - 1;
        },
        "minmax_element");

    run_algo_all_policies<read_archetype, read_archetype_dc, 15>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::is_sorted(std::forward<decltype(policy)>(policy), view, read_comp{});
        },
        [](auto&&, bool res) { return res; }, "is_sorted");

    // The whole range is sorted, so the scan stops at its end.
    run_algo_all_policies<read_archetype, read_archetype_dc, 16>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::is_sorted_until(std::forward<decltype(policy)>(policy), view, read_comp{});
        },
        [](auto&& view, auto res) { return res == std::ranges::begin(view) + std::ranges::size(view); },
        "is_sorted_until");

    // is_heap and is_heap_until are constrained exactly like is_sorted, i.e. by
    // std::indirect_strict_weak_order on the projected iterator, so the element type stays
    // non-copyable here as well. The range is ascending, so it is not a max-heap and the heap property
    // already breaks at the first child.
    run_algo_all_policies<read_archetype, read_archetype_dc, 17>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::is_heap(std::forward<decltype(policy)>(policy), view, read_comp{});
        },
        [](auto&&, bool res) { return !res; }, "is_heap");

    run_algo_all_policies<read_archetype, read_archetype_dc, 18>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::is_heap_until(std::forward<decltype(policy)>(policy), view, read_comp{});
        },
        [](auto&& view, auto res) { return res == std::ranges::begin(view) + 1; }, "is_heap_until");

    run_algo_all_policies<read_archetype, read_archetype_dc, 19>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::adjacent_find(std::forward<decltype(policy)>(policy), view, read_binary_pred{});
        },
        [](auto&& view, auto res) { return res == std::ranges::begin(view) + std::ranges::size(view); },
        "adjacent_find");

    //----------------------------------------------------------------------------------------------
    // The same algorithms with callables taking their arguments by non-const reference.
    //----------------------------------------------------------------------------------------------
    run_algo_all_policies<read_archetype, read_archetype_dc, 20>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::for_each(std::forward<decltype(policy)>(policy), view, read_unary_fun_mut{});
        },
        [](auto&& view, auto res) { return res == std::ranges::begin(view) + std::ranges::size(view); },
        "for_each, non-const callable");

    run_algo_all_policies<read_archetype, read_archetype_dc, 21>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::find_if(std::forward<decltype(policy)>(policy), view, read_unary_pred_mut{});
        },
        [](auto&& view, auto res) { return res == std::ranges::begin(view); }, "find_if, non-const callable");

    run_algo_all_policies<read_archetype, read_archetype_dc, 22>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::find_if_not(std::forward<decltype(policy)>(policy), view, read_unary_pred_mut{});
        },
        [](auto&& view, auto res) { return res == std::ranges::begin(view) + 1; }, "find_if_not, non-const callable");

    // The last element whose value is divisible by three, and the last one whose value is not.
    run_algo_all_policies<read_archetype, read_archetype_dc, 23>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::find_last_if(std::forward<decltype(policy)>(policy), view, read_unary_pred_mut{});
        },
        [](auto&& view, auto res) {
            auto __n = (int)std::ranges::size(view);
            return std::ranges::begin(res) == std::ranges::begin(view) + (__n - 1) / 3 * 3;
        },
        "find_last_if, non-const callable");

    run_algo_all_policies<read_archetype, read_archetype_dc, 24>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::find_last_if_not(std::forward<decltype(policy)>(policy), view, read_unary_pred_mut{});
        },
        [](auto&& view, auto res) {
            auto __n = (int)std::ranges::size(view);
            return std::ranges::begin(res) == std::ranges::begin(view) + ((__n - 1) % 3 == 0 ? __n - 2 : __n - 1);
        },
        "find_last_if_not, non-const callable");

    run_algo_all_policies<read_archetype, read_archetype_dc, 25>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::any_of(std::forward<decltype(policy)>(policy), view, read_unary_pred_mut{});
        },
        [](auto&&, bool res) { return res; }, "any_of, non-const callable");

    // Every third element satisfies the predicate, so the range is neither all nor none of it, and it
    // is not partitioned either.
    run_algo_all_policies<read_archetype, read_archetype_dc, 26>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::all_of(std::forward<decltype(policy)>(policy), view, read_unary_pred_mut{});
        },
        [](auto&&, bool res) { return !res; }, "all_of, non-const callable");

    run_algo_all_policies<read_archetype, read_archetype_dc, 27>(
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
    // Only the device call is broken, so the host policies keep their branches compiled.
    {
        auto call = [](auto&& policy, auto&& view) {
            return dpl_ranges::is_partitioned(std::forward<decltype(policy)>(policy), view, read_unary_pred_mut{});
        };
        auto check = [](auto&&, bool res) { return !res; };

        run_algo_host_policies<read_archetype>(call, check, "is_partitioned, non-const callable");
#if TEST_DPCPP_BACKEND_PRESENT && !_TEST_CPP20_RANGES_BROKEN_REQUIRES_IS_PARTITIONED_HETERO
        run_algo_hetero_policies<read_archetype_dc, 28>(call, check, "is_partitioned, non-const callable");
#endif
    }

    run_algo_all_policies<read_archetype, read_archetype_dc, 29>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::count_if(std::forward<decltype(policy)>(policy), view, read_unary_pred_mut{});
        },
        [](auto&& view, auto res) {
            return res == (std::ranges::range_difference_t<decltype(view)>)((std::ranges::size(view) + 2) / 3);
        },
        "count_if, non-const callable");

    // The projection takes the element by non-const reference; the predicate sees its prvalue result.
    run_algo_all_policies<read_archetype, read_archetype_dc, 30>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::find_if(std::forward<decltype(policy)>(policy), view, read_proj_pred{}, read_proj_mut{});
        },
        [](auto&& view, auto res) { return res == std::ranges::begin(view); }, "find_if, non-const projection");

    run_algo_all_policies<read_archetype, read_archetype_dc, 31>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::count_if(std::forward<decltype(policy)>(policy), view, read_proj_pred{},
                                        read_proj_mut{});
        },
        [](auto&& view, auto res) {
            return res == (std::ranges::range_difference_t<decltype(view)>)((std::ranges::size(view) + 2) / 3);
        },
        "count_if, non-const projection");

    run_algo_all_policies<read_archetype, read_archetype_dc, 32>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::adjacent_find(std::forward<decltype(policy)>(policy), view, read_binary_pred_mut{});
        },
        [](auto&& view, auto res) { return res == std::ranges::begin(view) + std::ranges::size(view); },
        "adjacent_find, non-const callable");

    run_algo_all_policies<read_archetype, read_archetype_dc, 33>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::is_sorted(std::forward<decltype(policy)>(policy), view, read_comp_mut{});
        },
        [](auto&&, bool res) { return res; }, "is_sorted, non-const comparator");

    // The whole range is sorted, so the scan stops at its end.
    run_algo_all_policies<read_archetype, read_archetype_dc, 34>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::is_sorted_until(std::forward<decltype(policy)>(policy), view, read_comp_mut{});
        },
        [](auto&& view, auto res) { return res == std::ranges::begin(view) + std::ranges::size(view); },
        "is_sorted_until, non-const comparator");

    // KSATODO: std::indirect_strict_weak_order only requires the comparator to be invocable with
    // iter_reference_t<_It>, a non-const lvalue here, but the device path compares two const lvalues,
    // so the call does not compile:
    //  - algorithm_impl_hetero.h:1120,1124 - __is_heap_check::operator() is const and subscripts the
    //    accessor of a read-only all_view, so both elements reach the comparator as const lvalues;
    //  - utils.h:203 - __binary_op::operator() forwards them into std::invoke;
    //  - unseq_backend_sycl.h:555,557 - single_match_pred_by_idx passes the accessor on as const.
    {
        auto call = [](auto&& policy, auto&& view) {
            return dpl_ranges::is_heap(std::forward<decltype(policy)>(policy), view, read_comp_mut{});
        };
        auto check = [](auto&&, bool res) { return !res; };

        run_algo_host_policies<read_archetype>(call, check, "is_heap, non-const comparator");
#if TEST_DPCPP_BACKEND_PRESENT && !_TEST_CPP20_RANGES_BROKEN_REQUIRES_IS_HEAP_HETERO
        run_algo_hetero_policies<read_archetype_dc, 35>(call, check, "is_heap, non-const comparator");
#endif
    }

    // KSATODO: is_heap_until shares __is_heap_check with is_heap, so its device path compares two const
    // lvalues in exactly the same way, see the note above.
    {
        auto call = [](auto&& policy, auto&& view) {
            return dpl_ranges::is_heap_until(std::forward<decltype(policy)>(policy), view, read_comp_mut{});
        };
        auto check = [](auto&& view, auto res) { return res == std::ranges::begin(view) + 1; };

        run_algo_host_policies<read_archetype>(call, check, "is_heap_until, non-const comparator");
#if TEST_DPCPP_BACKEND_PRESENT && !_TEST_CPP20_RANGES_BROKEN_REQUIRES_IS_HEAP_UNTIL_HETERO
        run_algo_hetero_policies<read_archetype_dc, 36>(call, check, "is_heap_until, non-const comparator");
#endif
    }

    run_algo_all_policies<read_archetype, read_archetype_dc, 37>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::min_element(std::forward<decltype(policy)>(policy), view, read_comp_mut{});
        },
        [](auto&& view, auto res) { return res == std::ranges::begin(view); }, "min_element, non-const comparator");

    run_algo_all_policies<read_archetype, read_archetype_dc, 38>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::max_element(std::forward<decltype(policy)>(policy), view, read_comp_mut{});
        },
        [](auto&& view, auto res) { return res == std::ranges::begin(view) + std::ranges::size(view) - 1; },
        "max_element, non-const comparator");

    run_algo_all_policies<read_archetype, read_archetype_dc, 39>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::minmax_element(std::forward<decltype(policy)>(policy), view, read_comp_mut{});
        },
        [](auto&& view, auto res) {
            return res.min == std::ranges::begin(view) &&
                   res.max == std::ranges::begin(view) + std::ranges::size(view) - 1;
        },
        "minmax_element, non-const comparator");

    //----------------------------------------------------------------------------------------------
    // The same algorithms called without a callable at all, i.e. with the default std::ranges::less
    // and std::ranges::equal_to. The ordering and the equality then have to come from the element
    // type, so these calls use ordered_archetype and equality_archetype: they are the only ones which
    // instantiate the default comparator path of the implementation, e.g. __is_comp_ascending on the
    // device side.
    //----------------------------------------------------------------------------------------------
    run_algo_all_policies<ordered_archetype, ordered_archetype_dc, 40>(
        [](auto&& policy, auto&& view) { return dpl_ranges::is_sorted(std::forward<decltype(policy)>(policy), view); },
        [](auto&&, bool res) { return res; }, "is_sorted, default comparator");

    run_algo_all_policies<ordered_archetype, ordered_archetype_dc, 41>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::is_sorted_until(std::forward<decltype(policy)>(policy), view);
        },
        [](auto&& view, auto res) { return res == std::ranges::begin(view) + std::ranges::size(view); },
        "is_sorted_until, default comparator");

    // The range is ascending, so it is not a max-heap and the heap property breaks at the first child.
    run_algo_all_policies<ordered_archetype, ordered_archetype_dc, 42>(
        [](auto&& policy, auto&& view) { return dpl_ranges::is_heap(std::forward<decltype(policy)>(policy), view); },
        [](auto&&, bool res) { return !res; }, "is_heap, default comparator");

    run_algo_all_policies<ordered_archetype, ordered_archetype_dc, 43>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::is_heap_until(std::forward<decltype(policy)>(policy), view);
        },
        [](auto&& view, auto res) { return res == std::ranges::begin(view) + 1; },
        "is_heap_until, default comparator");

    run_algo_all_policies<ordered_archetype, ordered_archetype_dc, 44>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::min_element(std::forward<decltype(policy)>(policy), view);
        },
        [](auto&& view, auto res) { return res == std::ranges::begin(view); }, "min_element, default comparator");

    run_algo_all_policies<ordered_archetype, ordered_archetype_dc, 45>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::max_element(std::forward<decltype(policy)>(policy), view);
        },
        [](auto&& view, auto res) { return res == std::ranges::begin(view) + std::ranges::size(view) - 1; },
        "max_element, default comparator");

    run_algo_all_policies<ordered_archetype, ordered_archetype_dc, 46>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::minmax_element(std::forward<decltype(policy)>(policy), view);
        },
        [](auto&& view, auto res) {
            return res.min == std::ranges::begin(view) &&
                   res.max == std::ranges::begin(view) + std::ranges::size(view) - 1;
        },
        "minmax_element, default comparator");

    // adjacent_find defaults its predicate to std::ranges::equal_to, which needs the equality of the
    // element type and nothing else; all the values differ, so the scan reaches the end of the range.
    run_algo_all_policies<equality_archetype, equality_archetype_dc, 47>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::adjacent_find(std::forward<decltype(policy)>(policy), view);
        },
        [](auto&& view, auto res) { return res == std::ranges::begin(view) + std::ranges::size(view); },
        "adjacent_find, default predicate");

    //----------------------------------------------------------------------------------------------
    // The same algorithms over plain_archetype_view, i.e. over a range which does not derive from
    // std::ranges::view_interface and therefore has neither size(), operator[], empty(), front() nor
    // back(). No requires-clause of any algorithm asks a range for those members: random_access_range
    // and sized_range are satisfied through begin(), end() and the sized sentinel alone, so an
    // implementation which reaches for a member of the user range instead of going through
    // std::ranges::begin / end / size does not compile here.
    //
    // Only one call per pattern shape is run this way, here and in the cross, the permute and the write
    // test: how the user range is accessed is a property of the dispatch and of the pattern and not of
    // the individual algorithm. The values are the ones of the calls above, so what is new is the range
    // and nothing else.
    //----------------------------------------------------------------------------------------------
    run_algo_plain_all_policies<read_archetype, read_archetype_dc, 48>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::for_each(std::forward<decltype(policy)>(policy), view, read_unary_fun{});
        },
        [](auto&& view, auto res) { return res == std::ranges::begin(view) + std::ranges::size(view); },
        "for_each, plain range");

    run_algo_plain_all_policies<read_archetype, read_archetype_dc, 49>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::find_if(std::forward<decltype(policy)>(policy), view, read_unary_pred{});
        },
        [](auto&& view, auto res) { return res == std::ranges::begin(view); }, "find_if, plain range");

#endif //_ENABLE_STD_RANGES_TESTING

    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
