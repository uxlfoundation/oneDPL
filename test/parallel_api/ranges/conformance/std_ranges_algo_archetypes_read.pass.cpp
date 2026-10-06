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

    {
        auto call = [](auto&& policy, auto&& view) {
            return dpl_ranges::find_last_if(std::forward<decltype(policy)>(policy), view, read_unary_pred{});
        };
        auto check = [](auto&& view, auto res) {
            auto __n = (int)std::ranges::size(view);
            return std::ranges::begin(res) == std::ranges::begin(view) + (__n - 1) / 3 * 3;
        };

        run_algo_host_policies<read_archetype>(call, check, "find_last_if");
#if TEST_DPCPP_BACKEND_PRESENT && !_TEST_CPP20_RANGES_BROKEN_REQUIRES_FIND_LAST_IF_HETERO
        run_algo_hetero_policies<read_archetype_dc, 3>(call, check, "find_last_if");
#endif
    }

    {
        auto call = [](auto&& policy, auto&& view) {
            return dpl_ranges::find_last_if_not(std::forward<decltype(policy)>(policy), view, read_unary_pred{});
        };
        auto check = [](auto&& view, auto res) {
            auto __n = (int)std::ranges::size(view);
            return std::ranges::begin(res) == std::ranges::begin(view) + ((__n - 1) % 3 == 0 ? __n - 2 : __n - 1);
        };

        run_algo_host_policies<read_archetype>(call, check, "find_last_if_not");
#if TEST_DPCPP_BACKEND_PRESENT && !_TEST_CPP20_RANGES_BROKEN_REQUIRES_FIND_LAST_IF_NOT_HETERO
        run_algo_hetero_policies<read_archetype_dc, 4>(call, check, "find_last_if_not");
#endif
    }

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

    run_algo_all_policies<read_archetype, read_archetype_dc, 16>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::is_sorted_until(std::forward<decltype(policy)>(policy), view, read_comp{});
        },
        [](auto&& view, auto res) { return res == std::ranges::begin(view) + std::ranges::size(view); },
        "is_sorted_until");

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

    {
        auto call = [](auto&& policy, auto&& view) {
            return dpl_ranges::find_last_if(std::forward<decltype(policy)>(policy), view, read_unary_pred_mut{});
        };
        auto check = [](auto&& view, auto res) {
            auto __n = (int)std::ranges::size(view);
            return std::ranges::begin(res) == std::ranges::begin(view) + (__n - 1) / 3 * 3;
        };

        run_algo_host_policies<read_archetype>(call, check, "find_last_if, non-const callable");
#if TEST_DPCPP_BACKEND_PRESENT && !_TEST_CPP20_RANGES_BROKEN_REQUIRES_FIND_LAST_IF_HETERO
        run_algo_hetero_policies<read_archetype_dc, 23>(call, check, "find_last_if, non-const callable");
#endif
    }

    {
        auto call = [](auto&& policy, auto&& view) {
            return dpl_ranges::find_last_if_not(std::forward<decltype(policy)>(policy), view, read_unary_pred_mut{});
        };
        auto check = [](auto&& view, auto res) {
            auto __n = (int)std::ranges::size(view);
            return std::ranges::begin(res) == std::ranges::begin(view) + ((__n - 1) % 3 == 0 ? __n - 2 : __n - 1);
        };

        run_algo_host_policies<read_archetype>(call, check, "find_last_if_not, non-const callable");
#if TEST_DPCPP_BACKEND_PRESENT && !_TEST_CPP20_RANGES_BROKEN_REQUIRES_FIND_LAST_IF_NOT_HETERO
        run_algo_hetero_policies<read_archetype_dc, 24>(call, check, "find_last_if_not, non-const callable");
#endif
    }

    run_algo_all_policies<read_archetype, read_archetype_dc, 25>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::any_of(std::forward<decltype(policy)>(policy), view, read_unary_pred_mut{});
        },
        [](auto&&, bool res) { return res; }, "any_of, non-const callable");

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

    run_algo_all_policies<read_archetype, read_archetype_dc, 34>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::is_sorted_until(std::forward<decltype(policy)>(policy), view, read_comp_mut{});
        },
        [](auto&& view, auto res) { return res == std::ranges::begin(view) + std::ranges::size(view); },
        "is_sorted_until, non-const comparator");

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

    run_algo_all_policies<ordered_archetype, ordered_archetype_dc, 40>(
        [](auto&& policy, auto&& view) { return dpl_ranges::is_sorted(std::forward<decltype(policy)>(policy), view); },
        [](auto&&, bool res) { return res; }, "is_sorted, default comparator");

    run_algo_all_policies<ordered_archetype, ordered_archetype_dc, 41>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::is_sorted_until(std::forward<decltype(policy)>(policy), view);
        },
        [](auto&& view, auto res) { return res == std::ranges::begin(view) + std::ranges::size(view); },
        "is_sorted_until, default comparator");

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

    run_algo_all_policies<equality_archetype, equality_archetype_dc, 47>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::adjacent_find(std::forward<decltype(policy)>(policy), view);
        },
        [](auto&& view, auto res) { return res == std::ranges::begin(view) + std::ranges::size(view); },
        "adjacent_find, default predicate");

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
