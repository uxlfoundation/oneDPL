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

// Every algorithm below has to be callable with an archetype which satisfies exactly its declared
// constraints. A failure here means the implementation requires more from a user type than the
// requires-clause of the algorithm declares.
using read_view = archetypes::archetype_view<archetypes::read_archetype>;
using seq_policy = decltype(oneapi::dpl::execution::seq);

static_assert(std::invocable<decltype(dpl_ranges::for_each), seq_policy, read_view&, archetypes::read_unary_fun>);
static_assert(std::invocable<decltype(dpl_ranges::find_if), seq_policy, read_view&, archetypes::read_unary_pred>);
static_assert(std::invocable<decltype(dpl_ranges::find_if_not), seq_policy, read_view&, archetypes::read_unary_pred>);
static_assert(std::invocable<decltype(dpl_ranges::find_last_if), seq_policy, read_view&, archetypes::read_unary_pred>);
static_assert(std::invocable<decltype(dpl_ranges::find_last_if_not), seq_policy, read_view&,
                             archetypes::read_unary_pred>);
static_assert(std::invocable<decltype(dpl_ranges::any_of), seq_policy, read_view&, archetypes::read_unary_pred>);
static_assert(std::invocable<decltype(dpl_ranges::all_of), seq_policy, read_view&, archetypes::read_unary_pred>);
static_assert(std::invocable<decltype(dpl_ranges::none_of), seq_policy, read_view&, archetypes::read_unary_pred>);
static_assert(std::invocable<decltype(dpl_ranges::count_if), seq_policy, read_view&, archetypes::read_unary_pred>);
static_assert(std::invocable<decltype(dpl_ranges::is_partitioned), seq_policy, read_view&,
                             archetypes::read_unary_pred>);
static_assert(std::invocable<decltype(dpl_ranges::adjacent_find), seq_policy, read_view&,
                             archetypes::read_binary_pred>);
static_assert(std::invocable<decltype(dpl_ranges::is_sorted), seq_policy, read_view&, archetypes::read_comp>);
static_assert(std::invocable<decltype(dpl_ranges::is_sorted_until), seq_policy, read_view&, archetypes::read_comp>);
static_assert(std::invocable<decltype(dpl_ranges::min_element), seq_policy, read_view&, archetypes::read_comp>);
static_assert(std::invocable<decltype(dpl_ranges::max_element), seq_policy, read_view&, archetypes::read_comp>);
static_assert(std::invocable<decltype(dpl_ranges::minmax_element), seq_policy, read_view&, archetypes::read_comp>);

// The projection is allowed to return a completely unrelated type, so the algorithm must never
// apply the predicate to the raw element.
static_assert(std::invocable<decltype(dpl_ranges::find_if), seq_policy, read_view&, archetypes::read_proj_pred,
                             archetypes::read_proj>);
static_assert(std::invocable<decltype(dpl_ranges::count_if), seq_policy, read_view&, archetypes::read_proj_pred,
                             archetypes::read_proj>);

// The search value type of find/count/contains is unrelated to the element type.
using searchable_view = archetypes::archetype_view<archetypes::searchable_archetype>;

static_assert(std::invocable<decltype(dpl_ranges::find), seq_policy, searchable_view&,
                             const archetypes::search_value&>);
static_assert(std::invocable<decltype(dpl_ranges::find_last), seq_policy, searchable_view&,
                             const archetypes::search_value&>);
static_assert(std::invocable<decltype(dpl_ranges::count), seq_policy, searchable_view&,
                             const archetypes::search_value&>);

// Two-range algorithms only require the predicate to accept the two projected references; the two
// element types stay unrelated and neither of them is comparable with itself.
using lhs_view = archetypes::archetype_view<archetypes::lhs_archetype>;
using rhs_view = archetypes::archetype_view<archetypes::rhs_archetype>;

static_assert(std::invocable<decltype(dpl_ranges::equal), seq_policy, lhs_view&, rhs_view&, archetypes::cross_pred>);
static_assert(std::invocable<decltype(dpl_ranges::mismatch), seq_policy, lhs_view&, rhs_view&, archetypes::cross_pred>);
static_assert(std::invocable<decltype(dpl_ranges::search), seq_policy, lhs_view&, rhs_view&, archetypes::cross_pred>);
static_assert(std::invocable<decltype(dpl_ranges::find_end), seq_policy, lhs_view&, rhs_view&, archetypes::cross_pred>);
static_assert(std::invocable<decltype(dpl_ranges::find_first_of), seq_policy, lhs_view&, rhs_view&,
                             archetypes::cross_pred>);

} //namespace test_std_ranges
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
    run_algo_all_policies<read_archetype, 0>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::find_if(std::forward<decltype(policy)>(policy), view, read_unary_pred{});
        },
        [](auto&& view, auto res) { return res == std::ranges::begin(view); }, "find_if");

    run_algo_all_policies<read_archetype, 1>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::find_if_not(std::forward<decltype(policy)>(policy), view, read_unary_pred{});
        },
        [](auto&& view, auto res) { return res == std::ranges::begin(view) + 1; }, "find_if_not");

    run_algo_all_policies<read_archetype, 2>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::any_of(std::forward<decltype(policy)>(policy), view, read_unary_pred{});
        },
        [](auto&&, bool res) { return res; }, "any_of");

    run_algo_all_policies<read_archetype, 3>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::all_of(std::forward<decltype(policy)>(policy), view, read_unary_pred{});
        },
        [](auto&&, bool res) { return !res; }, "all_of");

    run_algo_all_policies<read_archetype, 4>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::count_if(std::forward<decltype(policy)>(policy), view, read_unary_pred{});
        },
        [](auto&& view, auto res) { return res == (std::ranges::range_difference_t<decltype(view)>)
                                                      ((std::ranges::size(view) + 2) / 3); }, "count_if");

    // The projection returns an unrelated prvalue type, so the predicate can only ever be applied to
    // the projected value.
    run_algo_all_policies<read_archetype, 5>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::count_if(std::forward<decltype(policy)>(policy), view, read_proj_pred{}, read_proj{});
        },
        [](auto&& view, auto res) { return res == (std::ranges::range_difference_t<decltype(view)>)
                                                      ((std::ranges::size(view) + 2) / 3); }, "count_if with proj");

    // TODO: min_element/max_element/minmax_element are not run here. Their requires-clause only asks
    // for std::indirect_strict_weak_order, but the vectorized implementation
    // (__unseq_backend::__simd_min_element) stores a copy of the element in its _ComplexType helper,
    // so a non-copyable element type does not compile with the unseq policies. The static_asserts
    // above still check that the call itself is well-formed.

    run_algo_all_policies<read_archetype, 6>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::is_sorted(std::forward<decltype(policy)>(policy), view, read_comp{});
        },
        [](auto&&, bool res) { return res; }, "is_sorted");

    run_algo_all_policies<read_archetype, 7>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::adjacent_find(std::forward<decltype(policy)>(policy), view, read_binary_pred{});
        },
        [](auto&& view, auto res) { return res == std::ranges::begin(view) + std::ranges::size(view); },
        "adjacent_find");

    // The search value type is unrelated to the element type.
    run_algo_all_policies<searchable_archetype, 8>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::find(std::forward<decltype(policy)>(policy), view, search_value{7});
        },
        [](auto&& view, auto res) { return res == std::ranges::begin(view) + 7; }, "find");

    run_algo_all_policies<searchable_archetype, 9>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::count(std::forward<decltype(policy)>(policy), view, search_value{7});
        },
        [](auto&&, auto res) { return res == 1; }, "count");

    // Two ranges of unrelated element types, compared only through the user predicate.
    run_algo2_all_policies<lhs_archetype, rhs_archetype, 10>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            return dpl_ranges::equal(std::forward<decltype(policy)>(policy), view1, view2, cross_pred{});
        },
        [](auto&&, auto&&, bool res) { return res; }, "equal");

    run_algo2_all_policies<lhs_archetype, rhs_archetype, 11>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            return dpl_ranges::mismatch(std::forward<decltype(policy)>(policy), view1, view2, cross_pred{});
        },
        [](auto&& view1, auto&& view2, auto res) {
            return res.in1 == std::ranges::begin(view1) + std::ranges::size(view1) &&
                   res.in2 == std::ranges::begin(view2) + std::ranges::size(view2);
        },
        "mismatch");

#endif //_ENABLE_STD_RANGES_TESTING

    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
