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

    constexpr int searched = 3;

    run_algo_all_policies<searchable_archetype, searchable_archetype_dc, 0>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::find(std::forward<decltype(policy)>(policy), view, search_value{searched});
        },
        [](auto&& view, auto res) { return res == std::ranges::begin(view) + searched; }, "find");

    {
        auto call = [](auto&& policy, auto&& view) {
            return dpl_ranges::find_last(std::forward<decltype(policy)>(policy), view, search_value{searched});
        };
        auto check = [](auto&& view, auto res) {
            return std::ranges::begin(res) == std::ranges::begin(view) + searched;
        };

        run_algo_host_policies<searchable_archetype>(call, check, "find_last");
#if TEST_DPCPP_BACKEND_PRESENT && !_TEST_CPP20_RANGES_BROKEN_REQUIRES_FIND_LAST_HETERO
        run_algo_hetero_policies<searchable_archetype_dc, 1>(call, check, "find_last");
#endif
    }

    run_algo_all_policies<searchable_archetype, searchable_archetype_dc, 2>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::count(std::forward<decltype(policy)>(policy), view, search_value{searched});
        },
        [](auto&&, auto res) { return res == 1; }, "count");

    run_algo_all_policies<searchable_archetype, searchable_archetype_dc, 3>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::contains(std::forward<decltype(policy)>(policy), view, search_value{searched});
        },
        [](auto&&, auto res) { return res; }, "contains");

    run_algo_all_policies<searchable_archetype, searchable_archetype_dc, 4>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::search_n(std::forward<decltype(policy)>(policy), view, 1, search_value{searched});
        },
        [](auto&& view, auto res) {
            return std::ranges::begin(res) == std::ranges::begin(view) + searched && std::ranges::size(res) == 1;
        },
        "search_n");

    run_algo_all_policies<removable_archetype, removable_archetype_dc, 5>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::remove(std::forward<decltype(policy)>(policy), view, search_value{searched});
        },
        [](auto&&, auto res) { return std::ranges::size(res) == 1; }, "remove");

    run_algo_all_policies<searchable_archetype, searchable_archetype_dc, 6>(
        [](auto&& policy, auto&& view) {
            using elem_t = std::ranges::range_value_t<std::remove_cvref_t<decltype(view)>>;
            return dpl_ranges::find(std::forward<decltype(policy)>(policy), view,
                                    typename elem_t::nocopy_value_type{searched});
        },
        [](auto&& view, auto res) { return res == std::ranges::begin(view) + searched; }, "find, noncopyable value");

    {
        auto call = [](auto&& policy, auto&& view) {
            using elem_t = std::ranges::range_value_t<std::remove_cvref_t<decltype(view)>>;
            return dpl_ranges::find_last(std::forward<decltype(policy)>(policy), view,
                                         typename elem_t::nocopy_value_type{searched});
        };
        auto check = [](auto&& view, auto res) {
            return std::ranges::begin(res) == std::ranges::begin(view) + searched;
        };

        run_algo_host_policies<searchable_archetype>(call, check, "find_last, noncopyable value");
#if TEST_DPCPP_BACKEND_PRESENT && !_TEST_CPP20_RANGES_BROKEN_REQUIRES_FIND_LAST_HETERO
        run_algo_hetero_policies<searchable_archetype_dc, 7>(call, check, "find_last, noncopyable value");
#endif
    }

    run_algo_all_policies<searchable_archetype, searchable_archetype_dc, 8>(
        [](auto&& policy, auto&& view) {
            using elem_t = std::ranges::range_value_t<std::remove_cvref_t<decltype(view)>>;
            return dpl_ranges::count(std::forward<decltype(policy)>(policy), view,
                                     typename elem_t::nocopy_value_type{searched});
        },
        [](auto&&, auto res) { return res == 1; }, "count, noncopyable value");

    run_algo_all_policies<searchable_archetype, searchable_archetype_dc, 9>(
        [](auto&& policy, auto&& view) {
            using elem_t = std::ranges::range_value_t<std::remove_cvref_t<decltype(view)>>;
            return dpl_ranges::contains(std::forward<decltype(policy)>(policy), view,
                                        typename elem_t::nocopy_value_type{searched});
        },
        [](auto&&, auto res) { return res; }, "contains, noncopyable value");

    run_algo_all_policies<searchable_archetype, searchable_archetype_dc, 10>(
        [](auto&& policy, auto&& view) {
            using elem_t = std::ranges::range_value_t<std::remove_cvref_t<decltype(view)>>;
            return dpl_ranges::search_n(std::forward<decltype(policy)>(policy), view, 1,
                                        typename elem_t::nocopy_value_type{searched});
        },
        [](auto&& view, auto res) {
            return std::ranges::begin(res) == std::ranges::begin(view) + searched && std::ranges::size(res) == 1;
        },
        "search_n, noncopyable value");

    run_algo_all_policies<removable_archetype, removable_archetype_dc, 11>(
        [](auto&& policy, auto&& view) {
            using elem_t = std::ranges::range_value_t<std::remove_cvref_t<decltype(view)>>;
            return dpl_ranges::remove(std::forward<decltype(policy)>(policy), view,
                                      typename elem_t::nocopy_value_type{searched});
        },
        [](auto&&, auto res) { return std::ranges::size(res) == 1; }, "remove, noncopyable value");

    run_algo_all_policies<searchable_archetype, searchable_archetype_dc, 12>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::find(std::forward<decltype(policy)>(policy), view, search_value{searched},
                                    search_proj_mut{});
        },
        [](auto&& view, auto res) { return res == std::ranges::begin(view) + searched; },
        "find, non-const projection");

    {
        auto call = [](auto&& policy, auto&& view) {
            return dpl_ranges::find_last(std::forward<decltype(policy)>(policy), view, search_value{searched},
                                         search_proj_mut{});
        };
        auto check = [](auto&& view, auto res) {
            return std::ranges::begin(res) == std::ranges::begin(view) + searched;
        };

        run_algo_host_policies<searchable_archetype>(call, check, "find_last, non-const projection");
#if TEST_DPCPP_BACKEND_PRESENT && !_TEST_CPP20_RANGES_BROKEN_REQUIRES_FIND_LAST_HETERO
        run_algo_hetero_policies<searchable_archetype_dc, 13>(call, check, "find_last, non-const projection");
#endif
    }

    run_algo_all_policies<searchable_archetype, searchable_archetype_dc, 14>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::count(std::forward<decltype(policy)>(policy), view, search_value{searched},
                                     search_proj_mut{});
        },
        [](auto&&, auto res) { return res == 1; }, "count, non-const projection");

    run_algo_all_policies<searchable_archetype, searchable_archetype_dc, 15>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::contains(std::forward<decltype(policy)>(policy), view, search_value{searched},
                                        search_proj_mut{});
        },
        [](auto&&, auto res) { return res; }, "contains, non-const projection");

    run_algo_all_policies<searchable_archetype, searchable_archetype_dc, 16>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::search_n(std::forward<decltype(policy)>(policy), view, 1, search_value{searched},
                                        std::ranges::equal_to{}, search_proj_mut{});
        },
        [](auto&& view, auto res) {
            return std::ranges::begin(res) == std::ranges::begin(view) + searched && std::ranges::size(res) == 1;
        },
        "search_n, non-const projection");

    run_algo_all_policies<removable_archetype, removable_archetype_dc, 17>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::remove(std::forward<decltype(policy)>(policy), view, search_value{searched},
                                      search_proj_mut{});
        },
        [](auto&&, auto res) { return std::ranges::size(res) == 1; }, "remove, non-const projection");

#endif //_ENABLE_STD_RANGES_TESTING

    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
