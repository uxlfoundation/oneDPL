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

namespace test_std_ranges
{
namespace dpl_ranges = oneapi::dpl::ranges;

using seq_policy = decltype(oneapi::dpl::execution::seq);

using permutable_view = archetypes::archetype_view<archetypes::permutable_archetype>;

// The permuting algorithms are constrained by std::permutable<iterator_t<_R>> only, which requires
// the element to be movable, but not copyable, not default constructible and not comparable: any
// ordering or equality has to come from the comparator passed by the user.
static_assert(std::invocable<decltype(dpl_ranges::reverse), seq_policy, permutable_view&>);
static_assert(std::invocable<decltype(dpl_ranges::remove_if), seq_policy, permutable_view&,
                             archetypes::permutable_pred>);
static_assert(std::invocable<decltype(dpl_ranges::unique), seq_policy, permutable_view&, archetypes::permutable_equiv>);
static_assert(std::invocable<decltype(dpl_ranges::partition), seq_policy, permutable_view&,
                             archetypes::permutable_pred>);

// std::sortable<It, _Comp, _Proj> == permutable<It> && indirect_strict_weak_order<...>, so the very
// same element archetype works and the ordering never comes from an operator< on the element.
static_assert(std::invocable<decltype(dpl_ranges::sort), seq_policy, permutable_view&, archetypes::permutable_comp>);
static_assert(std::invocable<decltype(dpl_ranges::stable_sort), seq_policy, permutable_view&,
                             archetypes::permutable_comp>);
static_assert(std::invocable<decltype(dpl_ranges::is_sorted), seq_policy, permutable_view&,
                             archetypes::permutable_comp>);

} //namespace test_std_ranges
#endif //_ENABLE_STD_RANGES_TESTING

int
main()
{
#if _ENABLE_STD_RANGES_TESTING
    using namespace test_std_ranges;
    using namespace test_std_ranges::archetypes;
    namespace dpl_ranges = oneapi::dpl::ranges;

    // permutable_archetype is movable but not copyable, so it is not device copyable either: the
    // host policies are the only ones its constraints allow.
    run_algo_all_policies<permutable_archetype, 0>(
        [](auto&& policy, auto&& view) { return dpl_ranges::reverse(std::forward<decltype(policy)>(policy), view); },
        [](auto&& view, auto) {
            const auto n = std::ranges::size(view);
            return std::ranges::begin(view)[0].val == (int)n - 1 && std::ranges::begin(view)[n - 1].val == 0;
        },
        "reverse");

    // The storage is filled with 0, 1, 2, ... so every third element is removed. The returned
    // subrange is the tail holding the removed elements.
    run_algo_host_policies<permutable_archetype>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::remove_if(std::forward<decltype(policy)>(policy), view, permutable_pred{});
        },
        [](auto&& view, auto res) {
            const auto n = std::ranges::size(view);
            return std::ranges::size(res) == (n + 2) / 3;
        },
        "remove_if");

#if !_TEST_CPP20_RANGES_BROKEN_REQUIRES_REMOVE_IF_HETERO
    // The storage is filled with 0, 1, 2, ... so every third element is removed. The returned
    // subrange is the tail holding the removed elements.
    run_algo_hetero_policies<permutable_archetype, 1>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::remove_if(std::forward<decltype(policy)>(policy), view, permutable_pred{});
        },
        [](auto&& view, auto res) {
            const auto n = std::ranges::size(view);
            return std::ranges::size(res) == (n + 2) / 3;
        },
        "remove_if");
#endif

    // All the elements are unique, so nothing is dropped.
    run_algo_host_policies<permutable_archetype>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::unique(std::forward<decltype(policy)>(policy), view, permutable_equiv{});
        },
        [](auto&& view, auto res) { return std::ranges::size(res) == 0; }, "unique");

#if !_TEST_CPP20_RANGES_BROKEN_REQUIRES_UNIQUE_HETERO
    // All the elements are unique, so nothing is dropped.
    run_algo_hetero_policies<permutable_archetype, 2>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::unique(std::forward<decltype(policy)>(policy), view, permutable_equiv{});
        },
        [](auto&& view, auto res) { return std::ranges::size(res) == 0; }, "unique");
#endif

    // prpbably incorrect type applied
    run_algo_host_policies<permutable_archetype>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::sort(std::forward<decltype(policy)>(policy), view, permutable_comp{});
        },
        [](auto&& view, auto) {
            return std::ranges::begin(view)[0].val == 0 &&
                   std::ranges::begin(view)[std::ranges::size(view) - 1].val == (int)std::ranges::size(view) - 1;
        },
        "sort");

#if !_TEST_CPP20_RANGES_BROKEN_REQUIRES_SORT_HETERO
    // prpbably incorrect type applied
    run_algo_hetero_policies<permutable_archetype, 3>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::sort(std::forward<decltype(policy)>(policy), view, permutable_comp{});
        },
        [](auto&& view, auto) {
            return std::ranges::begin(view)[0].val == 0 &&
                   std::ranges::begin(view)[std::ranges::size(view) - 1].val == (int)std::ranges::size(view) - 1;
        },
        "sort");
#endif

    // prpbably incorrect type applied
    run_algo_host_policies<permutable_archetype>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::stable_sort(std::forward<decltype(policy)>(policy), view, permutable_comp{});
        },
        [](auto&& view, auto) {
            return std::ranges::begin(view)[0].val == 0 &&
                   std::ranges::begin(view)[std::ranges::size(view) - 1].val == (int)std::ranges::size(view) - 1;
        },
        "stable_sort");

#if !_TEST_CPP20_RANGES_BROKEN_REQUIRES_STABLE_SORT_HETERO
    // prpbably incorrect type applied
    run_algo_hetero_policies<permutable_archetype, 4>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::stable_sort(std::forward<decltype(policy)>(policy), view, permutable_comp{});
        },
        [](auto&& view, auto) {
            return std::ranges::begin(view)[0].val == 0 &&
                   std::ranges::begin(view)[std::ranges::size(view) - 1].val == (int)std::ranges::size(view) - 1;
        },
        "stable_sort");
#endif

    run_algo_all_policies<permutable_archetype, 5>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::is_sorted(std::forward<decltype(policy)>(policy), view, permutable_comp{});
        },
        [](auto&&, auto res) { return res; }, "is_sorted");
#endif //_ENABLE_STD_RANGES_TESTING

    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
