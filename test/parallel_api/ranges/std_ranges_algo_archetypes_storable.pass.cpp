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

    // The storable archetype family: the algorithms which return an element by value and are therefore
    // constrained by std::indirectly_copyable_storable, i.e. min, max and minmax. Covers them first
    // with const comparators and then with comparators taking their arguments by non-const reference.

    run_algo_all_policies<storable_archetype, storable_archetype_dc, __LINE__>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::min(std::forward<decltype(policy)>(policy), view, storable_comp{});
        },
        [](auto&&, auto res) { return res.val == 0; }, "min");

    run_algo_all_policies<storable_archetype, storable_archetype_dc, __LINE__>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::max(std::forward<decltype(policy)>(policy), view, storable_comp{});
        },
        [](auto&&, auto res) { return res.val == (int)archetype_test_size - 1; }, "max");

    run_algo_all_policies<storable_archetype, storable_archetype_dc, __LINE__>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::minmax(std::forward<decltype(policy)>(policy), view, storable_comp{});
        },
        [](auto&&, auto&& res) { return res.min.val == 0 && res.max.val == (int)archetype_test_size - 1; }, "minmax");

    //----------------------------------------------------------------------------------------------
    // The same algorithms with callables taking their arguments by non-const reference.
    //----------------------------------------------------------------------------------------------
    run_algo_all_policies<storable_archetype, storable_archetype_dc, __LINE__>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::min(std::forward<decltype(policy)>(policy), view, storable_comp_mut{});
        },
        [](auto&&, auto res) { return res.val == 0; }, "min, non-const comparator");

    run_algo_all_policies<storable_archetype, storable_archetype_dc, __LINE__>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::max(std::forward<decltype(policy)>(policy), view, storable_comp_mut{});
        },
        [](auto&&, auto res) { return res.val == (int)archetype_test_size - 1; }, "max, non-const comparator");

    run_algo_all_policies<storable_archetype, storable_archetype_dc, __LINE__>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::minmax(std::forward<decltype(policy)>(policy), view, storable_comp_mut{});
        },
        [](auto&&, auto&& res) { return res.min.val == 0 && res.max.val == (int)archetype_test_size - 1; },
        "minmax, non-const comparator");

#endif //_ENABLE_STD_RANGES_TESTING

    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
