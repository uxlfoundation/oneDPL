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

using writable_view = archetypes::archetype_view<archetypes::writable_archetype>;
using copy_in_view = archetypes::archetype_view<archetypes::copy_in_archetype>;
using copy_out_view = archetypes::archetype_view<archetypes::copy_out_archetype>;
using move_in_view = archetypes::archetype_view<archetypes::move_in_archetype>;
using move_out_view = archetypes::archetype_view<archetypes::move_out_archetype>;
using swap_view = archetypes::archetype_view<archetypes::swap_archetype>;
using transform_in_view = archetypes::archetype_view<archetypes::transform_in_archetype>;
using transform_out_view = archetypes::archetype_view<archetypes::transform_out_archetype>;

// fill only requires std::indirectly_writable<iterator_t<_R>, const _T&>: the element type is not
// required to be copyable, movable or default constructible and _T stays unrelated to it.
static_assert(std::invocable<decltype(dpl_ranges::fill), seq_policy, writable_view&, const archetypes::write_value&>);

// The copying algorithms only require std::indirectly_copyable, so the output element is merely
// assignable from a non-const lvalue of the input element type.
static_assert(std::invocable<decltype(dpl_ranges::copy), seq_policy, copy_in_view&, copy_out_view&>);

// move requires std::indirectly_movable, which is strictly weaker: assigning from an lvalue is
// deliberately rejected by move_out_archetype, so an implementation copying instead of moving fails.
static_assert(std::invocable<decltype(dpl_ranges::move), seq_policy, move_in_view&, move_out_view&>);

// swap_ranges requires std::indirectly_swappable only, which the hidden friend swap provides
// without the element being move constructible or move assignable.
static_assert(std::invocable<decltype(dpl_ranges::swap_ranges), seq_policy, swap_view&, swap_view&>);

// transform writes the result of the functor, which is a third unrelated type; the functor itself
// only has to be std::copy_constructible.
static_assert(std::invocable<decltype(dpl_ranges::transform), seq_policy, transform_in_view&, transform_out_view&,
                             archetypes::transform_unary_op>);

} //namespace test_std_ranges
#endif //_ENABLE_STD_RANGES_TESTING

int
main()
{
#if _ENABLE_STD_RANGES_TESTING
    using namespace test_std_ranges;
    using namespace test_std_ranges::archetypes;
    namespace dpl_ranges = oneapi::dpl::ranges;

    // None of the archetypes below is device copyable, so the host policies are the only ones the
    // constraints of these algorithms allow.
    run_algo_host_policies<writable_archetype>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::fill(std::forward<decltype(policy)>(policy), view, write_value{42});
        },
        [](auto&& view, auto) {
            return std::ranges::begin(view)[0].val == 42 &&
                   std::ranges::begin(view)[std::ranges::size(view) - 1].val == 42;
        },
        "fill");

    run_algo2_host_policies<copy_in_archetype, copy_out_archetype>(
        [](auto&& policy, auto&& in_view, auto&& out_view) {
            return dpl_ranges::copy(std::forward<decltype(policy)>(policy), in_view, out_view);
        },
        [](auto&& in_view, auto&& out_view, auto) {
            return std::ranges::begin(out_view)[7].val == std::ranges::begin(in_view)[7].val;
        },
        "copy");

    run_algo2_host_policies<move_in_archetype, move_out_archetype>(
        [](auto&& policy, auto&& in_view, auto&& out_view) {
            return dpl_ranges::move(std::forward<decltype(policy)>(policy), in_view, out_view);
        },
        [](auto&&, auto&& out_view, auto) { return std::ranges::begin(out_view)[7].val == 7; }, "move");

    run_algo2_host_policies<swap_archetype, swap_archetype>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            return dpl_ranges::swap_ranges(std::forward<decltype(policy)>(policy), view1, view2);
        },
        [](auto&& view1, auto&& view2, auto) {
            return std::ranges::begin(view1)[7].val == 7 && std::ranges::begin(view2)[7].val == 7;
        },
        "swap_ranges");

    run_algo2_host_policies<transform_in_archetype, transform_out_archetype>(
        [](auto&& policy, auto&& in_view, auto&& out_view) {
            return dpl_ranges::transform(std::forward<decltype(policy)>(policy), in_view, out_view,
                                         transform_unary_op{});
        },
        [](auto&&, auto&& out_view, auto) { return std::ranges::begin(out_view)[7].val == 14; }, "transform");
#endif //_ENABLE_STD_RANGES_TESTING

    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
