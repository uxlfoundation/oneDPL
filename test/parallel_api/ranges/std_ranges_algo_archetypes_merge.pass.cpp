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

using merge_in_view = archetypes::archetype_view<archetypes::merge_in_archetype>;
using merge_out_view = archetypes::archetype_view<archetypes::merge_out_archetype>;
using storable_view = archetypes::archetype_view<archetypes::storable_archetype>;

// The merge family is constrained by std::mergeable, which asks for indirectly_copyable from both
// inputs into the output plus a strict weak order: the output element stays non-copyable itself and
// the ordering never comes from an operator< on the element.
static_assert(std::invocable<decltype(dpl_ranges::merge), seq_policy, merge_in_view&, merge_in_view&, merge_out_view&,
                             archetypes::merge_comp>);
static_assert(std::invocable<decltype(dpl_ranges::set_union), seq_policy, merge_in_view&, merge_in_view&,
                             merge_out_view&, archetypes::merge_comp>);
static_assert(std::invocable<decltype(dpl_ranges::set_intersection), seq_policy, merge_in_view&, merge_in_view&,
                             merge_out_view&, archetypes::merge_comp>);
static_assert(std::invocable<decltype(dpl_ranges::set_difference), seq_policy, merge_in_view&, merge_in_view&,
                             merge_out_view&, archetypes::merge_comp>);
static_assert(std::invocable<decltype(dpl_ranges::set_symmetric_difference), seq_policy, merge_in_view&,
                             merge_in_view&, merge_out_view&, archetypes::merge_comp>);

// min / max / minmax additionally require std::indirectly_copyable_storable<iterator_t<_R>,
// range_value_t<_R>*>, which does need a copy constructor and copy assignment, but still no default
// constructor and no ordering operator on the element.
static_assert(std::invocable<decltype(dpl_ranges::min), seq_policy, storable_view&, archetypes::storable_comp>);
static_assert(std::invocable<decltype(dpl_ranges::max), seq_policy, storable_view&, archetypes::storable_comp>);
static_assert(std::invocable<decltype(dpl_ranges::minmax), seq_policy, storable_view&, archetypes::storable_comp>);

} //namespace test_std_ranges
#endif //_ENABLE_STD_RANGES_TESTING

int
main()
{
#if _ENABLE_STD_RANGES_TESTING
    using namespace test_std_ranges;
    using namespace test_std_ranges::archetypes;
    namespace dpl_ranges = oneapi::dpl::ranges;

    // Neither archetype is device copyable, so the host policies are the only ones their
    // constraints allow.

    // Both inputs hold the very same sorted sequence 0, 1, 2, ...
    run_algo2_all_policies<merge_in_archetype, merge_in_archetype, 0>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            archetype_storage<merge_out_archetype, std::allocator<merge_out_archetype>> out_storage(
                std::allocator<merge_out_archetype>{}, 2 * archetype_test_size, [](std::size_t) { return 0; });
            auto out_view = out_storage.view();
            auto res = dpl_ranges::merge(std::forward<decltype(policy)>(policy), view1, view2, out_view, merge_comp{});
            return std::ranges::begin(out_view)[0].val == 0 && std::ranges::begin(out_view)[1].val == 0 &&
                   std::ranges::begin(out_view)[2].val == 1 && res.out == std::ranges::end(out_view);
        },
        [](auto&&, auto&&, auto res) { return res; }, "merge");

    // KSATODO: the set operations only require std::mergeable, i.e. indirectly_copyable from either
    // input into the output, which is an assignment and not a construction. The implementation
    // instead constructs the output element into raw memory, so the calls below do not compile:
    //  - set_algorithms_utils.h:91 - placement new of _OutValueType from *__it_in, which also takes
    //    the address of the element through std::addressof;
    //  - set_algorithms_utils.h:127,133,206 / memory_impl.h:96,111 - __uninitialized_copy_or_discard
    //    default constructs and copy constructs the output element type.
    // Fixing this means assigning through the output iterator instead of constructing in place.
    run_algo2_all_policies<merge_in_archetype, merge_in_archetype, 1>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            archetype_storage<merge_out_archetype, std::allocator<merge_out_archetype>> out_storage(
                std::allocator<merge_out_archetype>{}, 2 * archetype_test_size, [](std::size_t) { return 0; });
            auto out_view = out_storage.view();
            auto res =
                dpl_ranges::set_union(std::forward<decltype(policy)>(policy), view1, view2, out_view, merge_comp{});
            // The two inputs hold the very same sequence, so the union is that sequence itself.
            return std::ranges::begin(out_view)[7].val == 7 &&
                   (std::size_t)(res.out - std::ranges::begin(out_view)) == archetype_test_size;
        },
        [](auto&&, auto&&, auto res) { return res; }, "set_union");

    run_algo2_all_policies<merge_in_archetype, merge_in_archetype, 2>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            archetype_storage<merge_out_archetype, std::allocator<merge_out_archetype>> out_storage(
                std::allocator<merge_out_archetype>{}, 2 * archetype_test_size, [](std::size_t) { return 0; });
            auto out_view = out_storage.view();
            auto res = dpl_ranges::set_difference(std::forward<decltype(policy)>(policy), view1, view2, out_view,
                                                  merge_comp{});
            // The two inputs are equal, so the difference is empty.
            return res.out == std::ranges::begin(out_view);
        },
        [](auto&&, auto&&, auto res) { return res; }, "set_difference");

    // KSATODO: min / max / minmax only require std::indirectly_copyable_storable, which needs a copy
    // constructor and copy assignment, but no default constructor. The helpers of __simd_min_element
    // and __simd_minmax_element at unseq_backend_simd.h:635 and :695 value initialize their
    // _ValueType members in the default constructor, so the calls below do not compile with a
    // non-default-constructible element type.
    run_algo_all_policies<storable_archetype, 3>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::min(std::forward<decltype(policy)>(policy), view, storable_comp{});
        },
        [](auto&&, auto res) { return res.val == 0; }, "min");

    run_algo_all_policies<storable_archetype, 4>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::max(std::forward<decltype(policy)>(policy), view, storable_comp{});
        },
        [](auto&&, auto res) { return res.val == (int)archetype_test_size - 1; }, "max");

    run_algo_all_policies<storable_archetype, 5>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::minmax(std::forward<decltype(policy)>(policy), view, storable_comp{});
        },
        [](auto&&, auto&& res) { return res.min.val == 0 && res.max.val == (int)archetype_test_size - 1; }, "minmax");

#endif //_ENABLE_STD_RANGES_TESTING

    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
