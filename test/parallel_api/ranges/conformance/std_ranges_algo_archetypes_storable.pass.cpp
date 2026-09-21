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

    run_algo_all_policies<storable_archetype, storable_archetype_dc, 0>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::min(std::forward<decltype(policy)>(policy), view, storable_comp{});
        },
        [](auto&&, auto res) { return res.val == 0; }, "min");

    run_algo_all_policies<storable_archetype, storable_archetype_dc, 1>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::max(std::forward<decltype(policy)>(policy), view, storable_comp{});
        },
        [](auto&&, auto res) { return res.val == (int)archetype_test_size - 1; }, "max");

    run_algo_all_policies<storable_archetype, storable_archetype_dc, 2>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::minmax(std::forward<decltype(policy)>(policy), view, storable_comp{});
        },
        [](auto&&, auto&& res) { return res.min.val == 0 && res.max.val == (int)archetype_test_size - 1; }, "minmax");

#if !_TEST_CPP20_RANGES_BROKEN_REQUIRES_MIN_HOST || TEST_DPCPP_BACKEND_PRESENT
    {
        auto call = [](auto&& policy, auto&& view) {
            return dpl_ranges::min(std::forward<decltype(policy)>(policy), view, storable_comp_mut{});
        };
        auto check = [](auto&&, auto res) { return res.val == 0; };

#if !_TEST_CPP20_RANGES_BROKEN_REQUIRES_MIN_HOST
        run_algo_host_policies<storable_archetype>(call, check, "min, non-const comparator");
#endif
#if TEST_DPCPP_BACKEND_PRESENT
        run_algo_hetero_policies<storable_archetype_dc, 3>(call, check, "min, non-const comparator");
#endif
    }
#endif // !_TEST_CPP20_RANGES_BROKEN_REQUIRES_MIN_HOST || TEST_DPCPP_BACKEND_PRESENT

#if !_TEST_CPP20_RANGES_BROKEN_REQUIRES_MAX_HOST || TEST_DPCPP_BACKEND_PRESENT
    {
        auto call = [](auto&& policy, auto&& view) {
            return dpl_ranges::max(std::forward<decltype(policy)>(policy), view, storable_comp_mut{});
        };
        auto check = [](auto&&, auto res) { return res.val == (int)archetype_test_size - 1; };

#if !_TEST_CPP20_RANGES_BROKEN_REQUIRES_MAX_HOST
        run_algo_host_policies<storable_archetype>(call, check, "max, non-const comparator");
#endif
#if TEST_DPCPP_BACKEND_PRESENT
        run_algo_hetero_policies<storable_archetype_dc, 4>(call, check, "max, non-const comparator");
#endif
    }
#endif // !_TEST_CPP20_RANGES_BROKEN_REQUIRES_MAX_HOST || TEST_DPCPP_BACKEND_PRESENT

#if !_TEST_CPP20_RANGES_BROKEN_REQUIRES_MINMAX_HOST || TEST_DPCPP_BACKEND_PRESENT
    {
        auto call = [](auto&& policy, auto&& view) {
            return dpl_ranges::minmax(std::forward<decltype(policy)>(policy), view, storable_comp_mut{});
        };
        auto check = [](auto&&, auto&& res) {
            return res.min.val == 0 && res.max.val == (int)archetype_test_size - 1;
        };

#if !_TEST_CPP20_RANGES_BROKEN_REQUIRES_MINMAX_HOST
        run_algo_host_policies<storable_archetype>(call, check, "minmax, non-const comparator");
#endif
#if TEST_DPCPP_BACKEND_PRESENT
        run_algo_hetero_policies<storable_archetype_dc, 5>(call, check, "minmax, non-const comparator");
#endif
    }
#endif // !_TEST_CPP20_RANGES_BROKEN_REQUIRES_MINMAX_HOST || TEST_DPCPP_BACKEND_PRESENT

    run_algo_all_policies<storable_ordered_archetype, storable_ordered_archetype_dc, 6>(
        [](auto&& policy, auto&& view) { return dpl_ranges::min(std::forward<decltype(policy)>(policy), view); },
        [](auto&&, auto res) { return res.val == 0; }, "min, default comparator");

    run_algo_all_policies<storable_ordered_archetype, storable_ordered_archetype_dc, 7>(
        [](auto&& policy, auto&& view) { return dpl_ranges::max(std::forward<decltype(policy)>(policy), view); },
        [](auto&&, auto res) { return res.val == (int)archetype_test_size - 1; }, "max, default comparator");

    run_algo_all_policies<storable_ordered_archetype, storable_ordered_archetype_dc, 8>(
        [](auto&& policy, auto&& view) { return dpl_ranges::minmax(std::forward<decltype(policy)>(policy), view); },
        [](auto&&, auto&& res) { return res.min.val == 0 && res.max.val == (int)archetype_test_size - 1; },
        "minmax, default comparator");

#endif //_ENABLE_STD_RANGES_TESTING

    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
