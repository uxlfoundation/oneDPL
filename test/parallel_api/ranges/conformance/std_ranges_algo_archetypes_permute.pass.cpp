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

    {
        auto call = [](auto&& policy, auto&& view) {
            return dpl_ranges::reverse(std::forward<decltype(policy)>(policy), view);
        };
        auto check = [](auto&& view, auto) {
            const auto n = std::ranges::size(view);
            return std::ranges::begin(view)[0].val == (int)n - 1 && std::ranges::begin(view)[n - 1].val == 0;
        };

        run_algo_host_policies<permutable_archetype>(call, check, "reverse");
#if TEST_DPCPP_BACKEND_PRESENT && !_TEST_CPP20_RANGES_BROKEN_REQUIRES_REVERSE_HETERO
        run_algo_hetero_policies<permutable_archetype_dc, 0>(call, check, "reverse");
#endif
    }

    run_algo_all_policies<permutable_archetype, permutable_archetype_dc, 1>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::rotate(std::forward<decltype(policy)>(policy), view, std::ranges::begin(view) + 10);
        },
        [](auto&& view, auto) {
            return std::ranges::begin(view)[0].val == 10 &&
                    std::ranges::begin(view)[std::ranges::size(view) - 10].val == 0;
        },
        "rotate");

    run_algo_all_policies<permutable_archetype, permutable_archetype_dc, 2>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::shift_left(std::forward<decltype(policy)>(policy), view, 10);
        },
        [](auto&& view, auto) { return std::ranges::begin(view)[0].val == 10; }, "shift_left");

    run_algo_all_policies<permutable_archetype, permutable_archetype_dc, 3>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::shift_right(std::forward<decltype(policy)>(policy), view, 10);
        },
        [](auto&& view, auto) { return std::ranges::begin(view)[10].val == 0; }, "shift_right");

    run_algo_all_policies<permutable_archetype, permutable_archetype_dc, 4>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::remove_if(std::forward<decltype(policy)>(policy), view, permutable_pred{});
        },
        [](auto&& view, auto res) {
            const auto n = std::ranges::size(view);
            return std::ranges::size(res) == (n + 2) / 3;
        },
        "remove_if");

    run_algo_all_policies<permutable_archetype, permutable_archetype_dc, 5>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::unique(std::forward<decltype(policy)>(policy), view, permutable_equiv{});
        },
        [](auto&& view, auto res) { return std::ranges::size(res) == 0; }, "unique");

    run_algo_all_policies<permutable_archetype, permutable_archetype_dc, 6>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::partition(std::forward<decltype(policy)>(policy), view, permutable_pred{});
        },
        [](auto&& view, auto res) {
            return std::ranges::size(res) == std::ranges::size(view) - (std::ranges::size(view) + 2) / 3;
        },
        "partition");

    run_algo_all_policies<permutable_archetype, permutable_archetype_dc, 7>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::stable_partition(std::forward<decltype(policy)>(policy), view, permutable_pred{});
        },
        [](auto&& view, auto res) {
            return std::ranges::size(res) == std::ranges::size(view) - (std::ranges::size(view) + 2) / 3;
        },
        "stable_partition");

#if !_TEST_CPP20_RANGES_BROKEN_REQUIRES_SORT
    run_algo_all_policies<permutable_archetype, permutable_archetype_dc, 8>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::sort(std::forward<decltype(policy)>(policy), view, permutable_comp{});
        },
        [](auto&& view, auto) {
            return std::ranges::begin(view)[0].val == 0 &&
                    std::ranges::begin(view)[std::ranges::size(view) - 1].val == (int)std::ranges::size(view) - 1;
        },
        "sort");
#endif

#if !_TEST_CPP20_RANGES_BROKEN_REQUIRES_STABLE_SORT
    run_algo_all_policies<permutable_archetype, permutable_archetype_dc, 9>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::stable_sort(std::forward<decltype(policy)>(policy), view, permutable_comp{});
        },
        [](auto&& view, auto) {
            return std::ranges::begin(view)[0].val == 0 &&
                    std::ranges::begin(view)[std::ranges::size(view) - 1].val == (int)std::ranges::size(view) - 1;
        },
        "stable_sort");
#endif

    run_algo_all_policies<permutable_archetype, permutable_archetype_dc, 10>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::is_sorted(std::forward<decltype(policy)>(policy), view, permutable_comp{});
        },
        [](auto&&, auto res) { return res; }, "is_sorted");

#if !_TEST_CPP20_RANGES_BROKEN_REQUIRES_PARTIAL_SORT_HOST || TEST_DPCPP_BACKEND_PRESENT
    {
        auto call = [](auto&& policy, auto&& view) {
            return dpl_ranges::partial_sort(std::forward<decltype(policy)>(policy), view, std::ranges::begin(view) + 10,
                                            permutable_comp{});
        };
        auto check = [](auto&& view, auto) {
            return std::ranges::begin(view)[0].val == 0 && std::ranges::begin(view)[9].val == 9;
        };

#if !_TEST_CPP20_RANGES_BROKEN_REQUIRES_PARTIAL_SORT_HOST
        run_algo_host_policies<permutable_archetype>(call, check, "partial_sort");
#endif
#if TEST_DPCPP_BACKEND_PRESENT
        run_algo_hetero_policies<permutable_archetype_dc, 11>(call, check, "partial_sort");
#endif
    }
#endif // !_TEST_CPP20_RANGES_BROKEN_REQUIRES_PARTIAL_SORT_HOST || TEST_DPCPP_BACKEND_PRESENT

#if !_TEST_CPP20_RANGES_BROKEN_REQUIRES_PARTIAL_SORT_COPY
    run_algo_all_policies<psort_copy_in_archetype, psort_copy_in_archetype_dc, 12>(
        [](auto&& policy, auto&& view) {
            using elem_t = std::ranges::range_value_t<std::remove_cvref_t<decltype(view)>>;
            auto out_storage = make_out_storage<typename elem_t::out_type>(policy, 10);
            auto out_view = out_storage.view();
            auto res = dpl_ranges::partial_sort_copy(std::forward<decltype(policy)>(policy), view, out_view,
                                                        psort_copy_comp{});
            return std::ranges::begin(out_view)[0].val == (int)archetype_test_size - 1 &&
                    std::ranges::begin(out_view)[9].val == (int)archetype_test_size - 10 &&
                    res.out == std::ranges::end(out_view);
        },
        [](auto&&, auto res) { return res; }, "partial_sort_copy");
#endif

    run_algo_all_policies<permutable_archetype, permutable_archetype_dc, 13>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::nth_element(std::forward<decltype(policy)>(policy), view, std::ranges::begin(view) + 10,
                                            permutable_comp{});
        },
        [](auto&& view, auto) { return std::ranges::begin(view)[10].val == 10; }, "nth_element");

#if !_TEST_CPP20_RANGES_BROKEN_REQUIRES_INPLACE_MERGE
    run_algo_all_policies<permutable_archetype, permutable_archetype_dc, 14>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::inplace_merge(std::forward<decltype(policy)>(policy), view,
                                                std::ranges::begin(view) + std::ranges::size(view) / 2,
                                                permutable_comp{});
        },
        [](auto&& view, auto) {
            return std::ranges::begin(view)[0].val == 0 &&
                    std::ranges::begin(view)[std::ranges::size(view) - 1].val == (int)std::ranges::size(view) - 1;
        },
        "inplace_merge");
#endif

    run_algo_all_policies<permutable_archetype, permutable_archetype_dc, 15>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::remove_if(std::forward<decltype(policy)>(policy), view, permutable_pred_mut{});
        },
        [](auto&& view, auto res) { return std::ranges::size(res) == (std::ranges::size(view) + 2) / 3; },
        "remove_if, non-const callable");

    run_algo_all_policies<permutable_archetype, permutable_archetype_dc, 16>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::unique(std::forward<decltype(policy)>(policy), view, permutable_equiv_mut{});
        },
        [](auto&&, auto res) { return std::ranges::size(res) == 0; }, "unique, non-const callable");

    {
        auto call = [](auto&& policy, auto&& view) {
            return dpl_ranges::partition(std::forward<decltype(policy)>(policy), view, permutable_pred_mut{});
        };
        auto check = [](auto&& view, auto res) {
            return std::ranges::size(res) == std::ranges::size(view) - (std::ranges::size(view) + 2) / 3;
        };

        run_algo_host_policies<permutable_archetype>(call, check, "partition, non-const callable");
#if TEST_DPCPP_BACKEND_PRESENT && !_TEST_CPP20_RANGES_BROKEN_REQUIRES_PARTITION_HETERO
        run_algo_hetero_policies<permutable_archetype_dc, 17>(call, check, "partition, non-const callable");
#endif
    }

    {
        auto call = [](auto&& policy, auto&& view) {
            return dpl_ranges::stable_partition(std::forward<decltype(policy)>(policy), view, permutable_pred_mut{});
        };
        auto check = [](auto&& view, auto res) {
            return std::ranges::size(res) == std::ranges::size(view) - (std::ranges::size(view) + 2) / 3;
        };

        run_algo_host_policies<permutable_archetype>(call, check, "stable_partition, non-const callable");
#if TEST_DPCPP_BACKEND_PRESENT && !_TEST_CPP20_RANGES_BROKEN_REQUIRES_STABLE_PARTITION_HETERO
        run_algo_hetero_policies<permutable_archetype_dc, 18>(call, check, "stable_partition, non-const callable");
#endif
    }

    run_algo_all_policies<permutable_archetype, permutable_archetype_dc, 19>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::is_sorted(std::forward<decltype(policy)>(policy), view, permutable_comp_mut{});
        },
        [](auto&&, bool res) { return res; }, "is_sorted of a permutable range, non-const comparator");

#if !_TEST_CPP20_RANGES_BROKEN_REQUIRES_SORT
    run_algo_all_policies<permutable_archetype, permutable_archetype_dc, 20>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::sort(std::forward<decltype(policy)>(policy), view, permutable_comp_mut{});
        },
        [](auto&& view, auto) {
            return std::ranges::begin(view)[0].val == 0 &&
                    std::ranges::begin(view)[std::ranges::size(view) - 1].val == (int)std::ranges::size(view) - 1;
        },
        "sort, non-const comparator");
#endif

#if !_TEST_CPP20_RANGES_BROKEN_REQUIRES_STABLE_SORT
    run_algo_all_policies<permutable_archetype, permutable_archetype_dc, 21>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::stable_sort(std::forward<decltype(policy)>(policy), view, permutable_comp_mut{});
        },
        [](auto&& view, auto) {
            return std::ranges::begin(view)[0].val == 0 &&
                    std::ranges::begin(view)[std::ranges::size(view) - 1].val == (int)std::ranges::size(view) - 1;
        },
        "stable_sort, non-const comparator");
#endif

#if !_TEST_CPP20_RANGES_BROKEN_REQUIRES_PARTIAL_SORT_HOST || TEST_DPCPP_BACKEND_PRESENT
    {
        auto call = [](auto&& policy, auto&& view) {
            return dpl_ranges::partial_sort(std::forward<decltype(policy)>(policy), view, std::ranges::begin(view) + 10,
                                            permutable_comp_mut{});
        };
        auto check = [](auto&& view, auto) {
            return std::ranges::begin(view)[0].val == 0 && std::ranges::begin(view)[9].val == 9;
        };

#if !_TEST_CPP20_RANGES_BROKEN_REQUIRES_PARTIAL_SORT_HOST
        run_algo_host_policies<permutable_archetype>(call, check, "partial_sort, non-const comparator");
#endif
#if TEST_DPCPP_BACKEND_PRESENT
        run_algo_hetero_policies<permutable_archetype_dc, 22>(call, check, "partial_sort, non-const comparator");
#endif
    }
#endif // !_TEST_CPP20_RANGES_BROKEN_REQUIRES_PARTIAL_SORT_HOST || TEST_DPCPP_BACKEND_PRESENT

#if !_TEST_CPP20_RANGES_BROKEN_REQUIRES_PARTIAL_SORT_COPY
    run_algo_all_policies<psort_copy_in_archetype, psort_copy_in_archetype_dc, 23>(
        [](auto&& policy, auto&& view) {
            using elem_t = std::ranges::range_value_t<std::remove_cvref_t<decltype(view)>>;
            auto out_storage = make_out_storage<typename elem_t::out_type>(policy, 10);
            auto out_view = out_storage.view();
            auto res = dpl_ranges::partial_sort_copy(std::forward<decltype(policy)>(policy), view, out_view,
                                                        psort_copy_comp_mut{}, psort_copy_in_proj_mut{},
                                                        psort_copy_out_proj_mut{});
            return std::ranges::begin(out_view)[0].val == (int)archetype_test_size - 1 &&
                    std::ranges::begin(out_view)[9].val == (int)archetype_test_size - 10 &&
                    res.out == std::ranges::end(out_view);
        },
        [](auto&&, auto res) { return res; }, "partial_sort_copy, non-const callables");
#endif

#if !_TEST_CPP20_RANGES_BROKEN_REQUIRES_NTH_ELEMENT_HOST || TEST_DPCPP_BACKEND_PRESENT
    {
        auto call = [](auto&& policy, auto&& view) {
            return dpl_ranges::nth_element(std::forward<decltype(policy)>(policy), view, std::ranges::begin(view) + 10,
                                            permutable_comp_mut{});
        };
        auto check = [](auto&& view, auto) { return std::ranges::begin(view)[10].val == 10; };

#if !_TEST_CPP20_RANGES_BROKEN_REQUIRES_NTH_ELEMENT_HOST
        run_algo_host_policies<permutable_archetype>(call, check, "nth_element, non-const comparator");
#endif
#if TEST_DPCPP_BACKEND_PRESENT
        run_algo_hetero_policies<permutable_archetype_dc, 24>(call, check, "nth_element, non-const comparator");
#endif
    }
#endif // !_TEST_CPP20_RANGES_BROKEN_REQUIRES_NTH_ELEMENT_HOST || TEST_DPCPP_BACKEND_PRESENT

#if !_TEST_CPP20_RANGES_BROKEN_REQUIRES_INPLACE_MERGE
    run_algo_all_policies<permutable_archetype, permutable_archetype_dc, 25>(
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
#endif

#if !_TEST_CPP20_RANGES_BROKEN_REQUIRES_SORT
    run_algo_all_policies<permutable_ordered_archetype, permutable_ordered_archetype_dc, 26>(
        [](auto&& policy, auto&& view) { return dpl_ranges::sort(std::forward<decltype(policy)>(policy), view); },
        [](auto&& view, auto) {
            return std::ranges::begin(view)[0].val == 0 &&
                    std::ranges::begin(view)[std::ranges::size(view) - 1].val == (int)std::ranges::size(view) - 1;
        },
        "sort, default comparator");
#endif

#if !_TEST_CPP20_RANGES_BROKEN_REQUIRES_STABLE_SORT
    run_algo_all_policies<permutable_ordered_archetype, permutable_ordered_archetype_dc, 27>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::stable_sort(std::forward<decltype(policy)>(policy), view);
        },
        [](auto&& view, auto) {
            return std::ranges::begin(view)[0].val == 0 &&
                    std::ranges::begin(view)[std::ranges::size(view) - 1].val == (int)std::ranges::size(view) - 1;
        },
        "stable_sort, default comparator");
#endif

#if !_TEST_CPP20_RANGES_BROKEN_REQUIRES_PARTIAL_SORT_HOST || TEST_DPCPP_BACKEND_PRESENT
    {
        auto call = [](auto&& policy, auto&& view) {
            return dpl_ranges::partial_sort(std::forward<decltype(policy)>(policy), view,
                                            std::ranges::begin(view) + 10);
        };
        auto check = [](auto&& view, auto) {
            return std::ranges::begin(view)[0].val == 0 && std::ranges::begin(view)[9].val == 9;
        };

#if !_TEST_CPP20_RANGES_BROKEN_REQUIRES_PARTIAL_SORT_HOST
        run_algo_host_policies<permutable_ordered_archetype>(call, check, "partial_sort, default comparator");
#endif
#if TEST_DPCPP_BACKEND_PRESENT
        run_algo_hetero_policies<permutable_ordered_archetype_dc, 28>(call, check, "partial_sort, default comparator");
#endif
    }
#endif // !_TEST_CPP20_RANGES_BROKEN_REQUIRES_PARTIAL_SORT_HOST || TEST_DPCPP_BACKEND_PRESENT

    run_algo_all_policies<permutable_ordered_archetype, permutable_ordered_archetype_dc, 29>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::nth_element(std::forward<decltype(policy)>(policy), view,
                                            std::ranges::begin(view) + 10);
        },
        [](auto&& view, auto) { return std::ranges::begin(view)[10].val == 10; },
        "nth_element, default comparator");

#if !_TEST_CPP20_RANGES_BROKEN_REQUIRES_INPLACE_MERGE
    run_algo_all_policies<permutable_ordered_archetype, permutable_ordered_archetype_dc, 30>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::inplace_merge(std::forward<decltype(policy)>(policy), view,
                                                std::ranges::begin(view) + std::ranges::size(view) / 2);
        },
        [](auto&& view, auto) {
            return std::ranges::begin(view)[0].val == 0 &&
                    std::ranges::begin(view)[std::ranges::size(view) - 1].val == (int)std::ranges::size(view) - 1;
        },
        "inplace_merge, default comparator");
#endif

    run_algo_all_policies<permutable_equality_archetype, permutable_equality_archetype_dc, 31>(
        [](auto&& policy, auto&& view) { return dpl_ranges::unique(std::forward<decltype(policy)>(policy), view); },
        [](auto&& view, auto res) { return std::ranges::size(res) == 0; }, "unique, default predicate");

#if !_TEST_CPP20_RANGES_BROKEN_REQUIRES_SORT
    run_algo_all_policies<permutable_archetype, permutable_archetype_dc, 32>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::sort(std::forward<decltype(policy)>(policy), view, std::ranges::less{},
                                    permutable_proj_key{});
        },
        [](auto&& view, auto) {
            return std::ranges::begin(view)[0].val == 0 &&
                    std::ranges::begin(view)[std::ranges::size(view) - 1].val == (int)std::ranges::size(view) - 1;
        },
        "sort, projected key");
#endif

#if !_TEST_CPP20_RANGES_BROKEN_REQUIRES_STABLE_SORT
    run_algo_all_policies<permutable_archetype, permutable_archetype_dc, 33>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::stable_sort(std::forward<decltype(policy)>(policy), view, std::ranges::less{},
                                            permutable_proj_key{});
        },
        [](auto&& view, auto) {
            return std::ranges::begin(view)[0].val == 0 &&
                    std::ranges::begin(view)[std::ranges::size(view) - 1].val == (int)std::ranges::size(view) - 1;
        },
        "stable_sort, projected key");
#endif

#if !_TEST_CPP20_RANGES_BROKEN_REQUIRES_SORT
    run_algo_plain_all_policies<permutable_archetype, permutable_archetype_dc, 34>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::sort(std::forward<decltype(policy)>(policy), view, permutable_comp{});
        },
        [](auto&& view, auto) {
            return std::ranges::begin(view)[0].val == 0 &&
                    std::ranges::begin(view)[std::ranges::size(view) - 1].val == (int)std::ranges::size(view) - 1;
        },
        "sort, plain range");
#endif

#endif //_ENABLE_STD_RANGES_TESTING

    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
