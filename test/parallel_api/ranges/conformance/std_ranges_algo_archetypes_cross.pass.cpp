// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) UXL Foundation Contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

    run_algo2_all_policies<lhs_archetype, rhs_archetype, lhs_archetype_dc, rhs_archetype_dc, 0>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            return dpl_ranges::equal(std::forward<decltype(policy)>(policy), view1, view2, cross_pred{});
        },
        [](auto&&, auto&&, bool res) { return res; }, "equal");

    run_algo2_all_policies<lhs_archetype, rhs_archetype, lhs_archetype_dc, rhs_archetype_dc, 1>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            return dpl_ranges::mismatch(std::forward<decltype(policy)>(policy), view1, view2, cross_pred{});
        },
        [](auto&& view1, auto&& view2, auto res) {
            return res.in1 == std::ranges::begin(view1) + std::ranges::size(view1) &&
                    res.in2 == std::ranges::begin(view2) + std::ranges::size(view2);
        },
        "mismatch");

    run_algo2_all_policies<lhs_archetype, rhs_archetype, lhs_archetype_dc, rhs_archetype_dc, 2>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            return dpl_ranges::search(std::forward<decltype(policy)>(policy), view1, view2, cross_pred{});
        },
        [](auto&& view1, auto&& view2, auto res) {
            return std::ranges::begin(res) == std::ranges::begin(view1) &&
                    std::ranges::size(res) == std::ranges::size(view2);
        },
        "search");

    run_algo2_all_policies<lhs_archetype, rhs_archetype, lhs_archetype_dc, rhs_archetype_dc, 3>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            return dpl_ranges::find_end(std::forward<decltype(policy)>(policy), view1, view2, cross_pred{});
        },
        [](auto&& view1, auto&& view2, auto res) {
            return std::ranges::begin(res) == std::ranges::begin(view1) &&
                    std::ranges::size(res) == std::ranges::size(view2);
        },
        "find_end");

    run_algo2_all_policies<lhs_archetype, rhs_archetype, lhs_archetype_dc, rhs_archetype_dc, 4>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            return dpl_ranges::find_first_of(std::forward<decltype(policy)>(policy), view1, view2, cross_pred{});
        },
        [](auto&& view1, auto&&, auto res) { return res == std::ranges::begin(view1); }, "find_first_of");

    run_algo2_all_policies<lhs_archetype, rhs_archetype, lhs_archetype_dc, rhs_archetype_dc, 5>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            return dpl_ranges::includes(std::forward<decltype(policy)>(policy), view1, view2, cross_comp{});
        },
        [](auto&&, auto&&, bool res) { return res; }, "includes");

    run_algo2_all_policies<lhs_archetype, rhs_archetype, lhs_archetype_dc, rhs_archetype_dc, 6>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            return dpl_ranges::contains_subrange(std::forward<decltype(policy)>(policy), view1, view2, cross_pred{});
        },
        [](auto&&, auto&&, bool res) { return res; }, "contains_subrange");

    run_algo2_all_policies<lhs_archetype, rhs_archetype, lhs_archetype_dc, rhs_archetype_dc, 7>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            return dpl_ranges::starts_with(std::forward<decltype(policy)>(policy), view1, view2, cross_pred{});
        },
        [](auto&&, auto&&, bool res) { return res; }, "starts_with");

    run_algo2_all_policies<lhs_archetype, rhs_archetype, lhs_archetype_dc, rhs_archetype_dc, 8>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            return dpl_ranges::ends_with(std::forward<decltype(policy)>(policy), view1, view2, cross_pred{});
        },
        [](auto&&, auto&&, bool res) { return res; }, "ends_with");

    run_algo2_all_policies<lhs_archetype, rhs_archetype, lhs_archetype_dc, rhs_archetype_dc, 9>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            return dpl_ranges::lexicographical_compare(std::forward<decltype(policy)>(policy), view1, view2,
                                                        cross_comp{});
        },
        [](auto&&, auto&&, bool res) { return !res; }, "lexicographical_compare");

    run_algo2_all_policies<lhs_archetype, rhs_archetype, lhs_archetype_dc, rhs_archetype_dc, 10>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            return dpl_ranges::equal(std::forward<decltype(policy)>(policy), view1, view2, cross_pred_mut{});
        },
        [](auto&&, auto&&, bool res) { return res; }, "equal, non-const callable");

    run_algo2_all_policies<lhs_archetype, rhs_archetype, lhs_archetype_dc, rhs_archetype_dc, 11>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            return dpl_ranges::mismatch(std::forward<decltype(policy)>(policy), view1, view2, cross_pred_mut{});
        },
        [](auto&& view1, auto&& view2, auto res) {
            return res.in1 == std::ranges::begin(view1) + std::ranges::size(view1) &&
                    res.in2 == std::ranges::begin(view2) + std::ranges::size(view2);
        },
        "mismatch, non-const callable");

    run_algo2_all_policies<lhs_archetype, rhs_archetype, lhs_archetype_dc, rhs_archetype_dc, 12>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            return dpl_ranges::search(std::forward<decltype(policy)>(policy), view1, view2, cross_pred_mut{});
        },
        [](auto&& view1, auto&& view2, auto res) {
            return std::ranges::begin(res) == std::ranges::begin(view1) &&
                    std::ranges::size(res) == std::ranges::size(view2);
        },
        "search, non-const callable");

    run_algo2_all_policies<lhs_archetype, rhs_archetype, lhs_archetype_dc, rhs_archetype_dc, 13>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            return dpl_ranges::find_end(std::forward<decltype(policy)>(policy), view1, view2, cross_pred_mut{});
        },
        [](auto&& view1, auto&& view2, auto res) {
            return std::ranges::begin(res) == std::ranges::begin(view1) &&
                    std::ranges::size(res) == std::ranges::size(view2);
        },
        "find_end, non-const callable");

    {
        auto call = [](auto&& policy, auto&& view1, auto&& view2) {
            return dpl_ranges::find_first_of(std::forward<decltype(policy)>(policy), view1, view2, cross_pred_mut{});
        };
        auto check = [](auto&& view1, auto&&, auto res) { return res == std::ranges::begin(view1); };

        run_algo2_host_policies<lhs_archetype, rhs_archetype>(call, check, "find_first_of, non-const callable");
#if TEST_DPCPP_BACKEND_PRESENT && !_TEST_CPP20_RANGES_BROKEN_REQUIRES_FIND_FIRST_OF_HETERO
        run_algo2_hetero_policies<lhs_archetype_dc, rhs_archetype_dc, 14>(call, check,
                                                                          "find_first_of, non-const callable");
#endif
    }

    run_algo2_all_policies<lhs_archetype, rhs_archetype, lhs_archetype_dc, rhs_archetype_dc, 15>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            return dpl_ranges::includes(std::forward<decltype(policy)>(policy), view1, view2, cross_comp_mut{});
        },
        [](auto&&, auto&&, bool res) { return res; }, "includes, non-const comparator");

    run_algo2_all_policies<lhs_archetype, rhs_archetype, lhs_archetype_dc, rhs_archetype_dc, 16>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            return dpl_ranges::contains_subrange(std::forward<decltype(policy)>(policy), view1, view2,
                                                    cross_pred_mut{});
        },
        [](auto&&, auto&&, bool res) { return res; }, "contains_subrange, non-const callable");

    run_algo2_all_policies<lhs_archetype, rhs_archetype, lhs_archetype_dc, rhs_archetype_dc, 17>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            return dpl_ranges::starts_with(std::forward<decltype(policy)>(policy), view1, view2, cross_pred_mut{});
        },
        [](auto&&, auto&&, bool res) { return res; }, "starts_with, non-const callable");

    run_algo2_all_policies<lhs_archetype, rhs_archetype, lhs_archetype_dc, rhs_archetype_dc, 18>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            return dpl_ranges::ends_with(std::forward<decltype(policy)>(policy), view1, view2, cross_pred_mut{});
        },
        [](auto&&, auto&&, bool res) { return res; }, "ends_with, non-const callable");

#if !_TEST_CPP20_RANGES_BROKEN_REQUIRES_LEXICOGRAPHICAL_COMPARE
    run_algo2_all_policies<lhs_archetype, rhs_archetype, lhs_archetype_dc, rhs_archetype_dc, 19>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            return dpl_ranges::lexicographical_compare(std::forward<decltype(policy)>(policy), view1, view2,
                                                        cross_comp_mut{});
        },
        [](auto&&, auto&&, bool res) { return !res; }, "lexicographical_compare, non-const comparator");
#endif

    run_algo2_all_policies<equality_archetype, equality_archetype, equality_archetype_dc, equality_archetype_dc, 20>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            return dpl_ranges::equal(std::forward<decltype(policy)>(policy), view1, view2);
        },
        [](auto&&, auto&&, bool res) { return res; }, "equal, default predicate");

    run_algo2_all_policies<equality_archetype, equality_archetype, equality_archetype_dc, equality_archetype_dc, 21>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            return dpl_ranges::mismatch(std::forward<decltype(policy)>(policy), view1, view2);
        },
        [](auto&& view1, auto&& view2, auto res) {
            return res.in1 == std::ranges::begin(view1) + std::ranges::size(view1) &&
                    res.in2 == std::ranges::begin(view2) + std::ranges::size(view2);
        },
        "mismatch, default predicate");

    run_algo2_all_policies<equality_archetype, equality_archetype, equality_archetype_dc, equality_archetype_dc, 22>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            return dpl_ranges::search(std::forward<decltype(policy)>(policy), view1, view2);
        },
        [](auto&& view1, auto&& view2, auto res) {
            return std::ranges::begin(res) == std::ranges::begin(view1) &&
                    std::ranges::size(res) == std::ranges::size(view2);
        },
        "search, default predicate");

    run_algo2_all_policies<equality_archetype, equality_archetype, equality_archetype_dc, equality_archetype_dc, 23>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            return dpl_ranges::find_end(std::forward<decltype(policy)>(policy), view1, view2);
        },
        [](auto&& view1, auto&& view2, auto res) {
            return std::ranges::begin(res) == std::ranges::begin(view1) &&
                    std::ranges::size(res) == std::ranges::size(view2);
        },
        "find_end, default predicate");

    run_algo2_all_policies<equality_archetype, equality_archetype, equality_archetype_dc, equality_archetype_dc, 24>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            return dpl_ranges::find_first_of(std::forward<decltype(policy)>(policy), view1, view2);
        },
        [](auto&& view1, auto&&, auto res) { return res == std::ranges::begin(view1); },
        "find_first_of, default predicate");

    run_algo2_all_policies<equality_archetype, equality_archetype, equality_archetype_dc, equality_archetype_dc, 25>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            return dpl_ranges::contains_subrange(std::forward<decltype(policy)>(policy), view1, view2);
        },
        [](auto&&, auto&&, bool res) { return res; }, "contains_subrange, default predicate");

    run_algo2_all_policies<equality_archetype, equality_archetype, equality_archetype_dc, equality_archetype_dc, 26>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            return dpl_ranges::starts_with(std::forward<decltype(policy)>(policy), view1, view2);
        },
        [](auto&&, auto&&, bool res) { return res; }, "starts_with, default predicate");

    run_algo2_all_policies<equality_archetype, equality_archetype, equality_archetype_dc, equality_archetype_dc, 27>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            return dpl_ranges::ends_with(std::forward<decltype(policy)>(policy), view1, view2);
        },
        [](auto&&, auto&&, bool res) { return res; }, "ends_with, default predicate");

    run_algo2_all_policies<ordered_archetype, ordered_archetype, ordered_archetype_dc, ordered_archetype_dc, 28>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            return dpl_ranges::includes(std::forward<decltype(policy)>(policy), view1, view2);
        },
        [](auto&&, auto&&, bool res) { return res; }, "includes, default comparator");

    run_algo2_all_policies<ordered_archetype, ordered_archetype, ordered_archetype_dc, ordered_archetype_dc, 29>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            return dpl_ranges::lexicographical_compare(std::forward<decltype(policy)>(policy), view1, view2);
        },
        [](auto&&, auto&&, bool res) { return !res; }, "lexicographical_compare, default comparator");

    run_algo2_plain_all_policies<lhs_archetype, rhs_archetype, lhs_archetype_dc, rhs_archetype_dc, 30>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            return dpl_ranges::equal(std::forward<decltype(policy)>(policy), view1, view2, cross_pred{});
        },
        [](auto&&, auto&&, bool res) { return res; }, "equal, plain ranges");
#endif //_ENABLE_STD_RANGES_TESTING

    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
