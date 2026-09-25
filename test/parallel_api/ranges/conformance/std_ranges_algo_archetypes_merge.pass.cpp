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

    run_algo2_all_policies<merge_in_archetype, merge_in_archetype, merge_in_archetype_dc, merge_in_archetype_dc, 0>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            using elem_t = std::ranges::range_value_t<std::remove_cvref_t<decltype(view1)>>;
            auto out_storage = make_out_storage<typename elem_t::out_type>(policy, 2 * archetype_test_size);
            auto out_view = out_storage.view();
            auto res = dpl_ranges::merge(std::forward<decltype(policy)>(policy), view1, view2, out_view, merge_comp{});
            return std::ranges::begin(out_view)[0].val == 0 && std::ranges::begin(out_view)[1].val == 0 &&
                   std::ranges::begin(out_view)[2].val == 1 && res.out == std::ranges::end(out_view);
        },
        [](auto&&, auto&&, auto res) { return res; }, "merge");

#if !_TEST_CPP20_RANGES_BROKEN_REQUIRES_SET_UNION
    run_algo2_all_policies<merge_in_archetype, merge_in_archetype, merge_in_archetype_dc, merge_in_archetype_dc, 1>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            using elem_t = std::ranges::range_value_t<std::remove_cvref_t<decltype(view1)>>;
            auto out_storage = make_out_storage<typename elem_t::out_type>(policy, 2 * archetype_test_size);
            auto out_view = out_storage.view();
            auto res =
                dpl_ranges::set_union(std::forward<decltype(policy)>(policy), view1, view2, out_view, merge_comp{});
            return std::ranges::begin(out_view)[7].val == 7 &&
                   (std::size_t)(res.out - std::ranges::begin(out_view)) == archetype_test_size;
        },
        [](auto&&, auto&&, auto res) { return res; }, "set_union");
#endif

#if !_TEST_CPP20_RANGES_BROKEN_REQUIRES_SET_DIFFERENCE
    run_algo2_all_policies<merge_in_archetype, merge_in_archetype, merge_in_archetype_dc, merge_in_archetype_dc, 2>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            using elem_t = std::ranges::range_value_t<std::remove_cvref_t<decltype(view1)>>;
            auto out_storage = make_out_storage<typename elem_t::out_type>(policy, 2 * archetype_test_size);
            auto out_view = out_storage.view();
            auto res = dpl_ranges::set_difference(std::forward<decltype(policy)>(policy), view1, view2, out_view,
                                                  merge_comp{});
            return res.out == std::ranges::begin(out_view);
        },
        [](auto&&, auto&&, auto res) { return res; }, "set_difference");
#endif

#if !_TEST_CPP20_RANGES_BROKEN_REQUIRES_SET_INTERSECTION
    run_algo2_all_policies<merge_in_archetype, merge_in_archetype, merge_in_archetype_dc, merge_in_archetype_dc, 3>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            using elem_t = std::ranges::range_value_t<std::remove_cvref_t<decltype(view1)>>;
            auto out_storage = make_out_storage<typename elem_t::out_type>(policy, 2 * archetype_test_size);
            auto out_view = out_storage.view();
            auto res = dpl_ranges::set_intersection(std::forward<decltype(policy)>(policy), view1, view2, out_view,
                                                    merge_comp{});
            return std::ranges::begin(out_view)[7].val == 7 &&
                   (std::size_t)(res.out - std::ranges::begin(out_view)) == archetype_test_size;
        },
        [](auto&&, auto&&, auto res) { return res; }, "set_intersection");
#endif

#if !_TEST_CPP20_RANGES_BROKEN_REQUIRES_SET_SYMMETRIC_DIFFERENCE
    run_algo2_all_policies<merge_in_archetype, merge_in_archetype, merge_in_archetype_dc, merge_in_archetype_dc, 4>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            using elem_t = std::ranges::range_value_t<std::remove_cvref_t<decltype(view1)>>;
            auto out_storage = make_out_storage<typename elem_t::out_type>(policy, 2 * archetype_test_size);
            auto out_view = out_storage.view();
            auto res = dpl_ranges::set_symmetric_difference(std::forward<decltype(policy)>(policy), view1, view2,
                                                            out_view, merge_comp{});
            return res.out == std::ranges::begin(out_view);
        },
        [](auto&&, auto&&, auto res) { return res; }, "set_symmetric_difference");
#endif

    run_algo2_all_policies<merge_in_archetype, merge_in_archetype, merge_in_archetype_dc, merge_in_archetype_dc, 5>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            using elem_t = std::ranges::range_value_t<std::remove_cvref_t<decltype(view1)>>;
            auto out_storage = make_out_storage<typename elem_t::out_type>(policy, 2 * archetype_test_size);
            auto out_view = out_storage.view();
            auto res =
                dpl_ranges::merge(std::forward<decltype(policy)>(policy), view1, view2, out_view, merge_comp_mut{});
            return std::ranges::begin(out_view)[0].val == 0 && std::ranges::begin(out_view)[1].val == 0 &&
                   std::ranges::begin(out_view)[2].val == 1 && res.out == std::ranges::end(out_view);
        },
        [](auto&&, auto&&, auto res) { return res; }, "merge, non-const comparator");

#if !_TEST_CPP20_RANGES_BROKEN_REQUIRES_SET_UNION
    run_algo2_all_policies<merge_in_archetype, merge_in_archetype, merge_in_archetype_dc, merge_in_archetype_dc, 6>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            using elem_t = std::ranges::range_value_t<std::remove_cvref_t<decltype(view1)>>;
            auto out_storage = make_out_storage<typename elem_t::out_type>(policy, 2 * archetype_test_size);
            auto out_view = out_storage.view();
            auto res = dpl_ranges::set_union(std::forward<decltype(policy)>(policy), view1, view2, out_view,
                                             merge_comp_mut{});
            return std::ranges::begin(out_view)[7].val == 7 &&
                   (std::size_t)(res.out - std::ranges::begin(out_view)) == archetype_test_size;
        },
        [](auto&&, auto&&, auto res) { return res; }, "set_union, non-const comparator");
#endif

#if !_TEST_CPP20_RANGES_BROKEN_REQUIRES_SET_DIFFERENCE
    run_algo2_all_policies<merge_in_archetype, merge_in_archetype, merge_in_archetype_dc, merge_in_archetype_dc, 7>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            using elem_t = std::ranges::range_value_t<std::remove_cvref_t<decltype(view1)>>;
            auto out_storage = make_out_storage<typename elem_t::out_type>(policy, 2 * archetype_test_size);
            auto out_view = out_storage.view();
            auto res = dpl_ranges::set_difference(std::forward<decltype(policy)>(policy), view1, view2, out_view,
                                                 merge_comp_mut{});
            return res.out == std::ranges::begin(out_view);
        },
        [](auto&&, auto&&, auto res) { return res; }, "set_difference, non-const comparator");
#endif

#if !_TEST_CPP20_RANGES_BROKEN_REQUIRES_SET_INTERSECTION
    run_algo2_all_policies<merge_in_archetype, merge_in_archetype, merge_in_archetype_dc, merge_in_archetype_dc, 8>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            using elem_t = std::ranges::range_value_t<std::remove_cvref_t<decltype(view1)>>;
            auto out_storage = make_out_storage<typename elem_t::out_type>(policy, 2 * archetype_test_size);
            auto out_view = out_storage.view();
            auto res = dpl_ranges::set_intersection(std::forward<decltype(policy)>(policy), view1, view2, out_view,
                                                   merge_comp_mut{});
            return std::ranges::begin(out_view)[7].val == 7 &&
                   (std::size_t)(res.out - std::ranges::begin(out_view)) == archetype_test_size;
        },
        [](auto&&, auto&&, auto res) { return res; }, "set_intersection, non-const comparator");
#endif

#if !_TEST_CPP20_RANGES_BROKEN_REQUIRES_SET_SYMMETRIC_DIFFERENCE
    run_algo2_all_policies<merge_in_archetype, merge_in_archetype, merge_in_archetype_dc, merge_in_archetype_dc, 9>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            using elem_t = std::ranges::range_value_t<std::remove_cvref_t<decltype(view1)>>;
            auto out_storage = make_out_storage<typename elem_t::out_type>(policy, 2 * archetype_test_size);
            auto out_view = out_storage.view();
            auto res = dpl_ranges::set_symmetric_difference(std::forward<decltype(policy)>(policy), view1, view2,
                                                            out_view, merge_comp_mut{});
            return res.out == std::ranges::begin(out_view);
        },
        [](auto&&, auto&&, auto res) { return res; }, "set_symmetric_difference, non-const comparator");
#endif

    run_algo2_all_policies<merge_ordered_in_archetype, merge_ordered_in_archetype, merge_ordered_in_archetype_dc,
                           merge_ordered_in_archetype_dc, 10>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            using elem_t = std::ranges::range_value_t<std::remove_cvref_t<decltype(view1)>>;
            auto out_storage = make_out_storage<typename elem_t::out_type>(policy, 2 * archetype_test_size);
            auto out_view = out_storage.view();
            auto res = dpl_ranges::merge(std::forward<decltype(policy)>(policy), view1, view2, out_view);
            return std::ranges::begin(out_view)[0].val == 0 && std::ranges::begin(out_view)[1].val == 0 &&
                   std::ranges::begin(out_view)[2].val == 1 && res.out == std::ranges::end(out_view);
        },
        [](auto&&, auto&&, auto res) { return res; }, "merge, default comparator");

#endif //_ENABLE_STD_RANGES_TESTING

    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
