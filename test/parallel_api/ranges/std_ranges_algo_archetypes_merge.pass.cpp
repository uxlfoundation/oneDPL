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
    run_algo2_host_policies<merge_in_archetype, merge_in_archetype>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            archetype_storage<merge_out_archetype, std::allocator<merge_out_archetype>> out_storage(
                std::allocator<merge_out_archetype>{}, 2 * archetype_test_size, [](std::size_t) { return 0; });
            auto out_view = out_storage.view();
            auto res = dpl_ranges::merge(std::forward<decltype(policy)>(policy), view1, view2, out_view, merge_comp{});
            return std::ranges::begin(out_view)[0].val == 0 && std::ranges::begin(out_view)[1].val == 0 &&
                   std::ranges::begin(out_view)[2].val == 1 && res.out == std::ranges::end(out_view);
        },
        [](auto&&, auto&&, auto res) { return res; }, "merge");

    // KSATODO: add runtime coverage of set_union / set_intersection / set_difference /
    // set_symmetric_difference. The current implementation default constructs and copy constructs
    // the output element in set_algorithms_utils.h, which std::mergeable does not ask for, so it
    // does not compile with merge_out_archetype.

    // KSATODO: add runtime coverage of min / max / minmax. The vectorized path in
    // unseq_backend_simd.h default constructs the element type, which
    // std::indirectly_copyable_storable does not ask for, so it does not compile with
    // storable_archetype.
#endif //_ENABLE_STD_RANGES_TESTING

    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
