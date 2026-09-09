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
#endif //_ENABLE_STD_RANGES_TESTING

int
main()
{
#if _ENABLE_STD_RANGES_TESTING
    using namespace test_std_ranges;
    using namespace test_std_ranges::archetypes;
    namespace dpl_ranges = oneapi::dpl::ranges;

    // This file covers the search value archetype family: searchable_archetype and
    // removable_archetype (with their device copyable _dc counterparts) as the element type, and
    // search_value / nocopy_search_value as the searched value. The algorithms are find, find_last,
    // count, contains and remove, both with const callables and with a non-const projection.

    // The storage is filled with the values 0, 1, 2, ... so the value 3 is found exactly once.
    constexpr int searched = 3;

    //----------------------------------------------------------------------------------------------
    // The value based algorithms: the search value is compared with std::ranges::equal_to, so the
    // value type itself is the only requirement beyond the element type.
    //----------------------------------------------------------------------------------------------
    // search_value is trivially copyable and thus device copyable, so it can be used with all the
    // policies including the device ones.
    run_algo_all_policies<searchable_archetype, searchable_archetype_dc, __LINE__>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::find(std::forward<decltype(policy)>(policy), view, search_value{searched});
        },
        [](auto&& view, auto res) { return res == std::ranges::begin(view) + searched; }, "find");

    run_algo_all_policies<searchable_archetype, searchable_archetype_dc, __LINE__>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::find_last(std::forward<decltype(policy)>(policy), view, search_value{searched});
        },
        [](auto&& view, auto res) { return std::ranges::begin(res) == std::ranges::begin(view) + searched; },
        "find_last");

    run_algo_all_policies<searchable_archetype, searchable_archetype_dc, __LINE__>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::count(std::forward<decltype(policy)>(policy), view, search_value{searched});
        },
        [](auto&&, auto res) { return res == 1; }, "count");

    run_algo_all_policies<searchable_archetype, searchable_archetype_dc, __LINE__>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::contains(std::forward<decltype(policy)>(policy), view, search_value{searched});
        },
        [](auto&&, auto res) { return res; }, "contains");

    // remove() moves the surviving elements over the removed ones, so its element type has to be
    // movable: removable_archetype adds a move constructor and move assignment to the searchable
    // archetype and nothing else.
    run_algo_all_policies<removable_archetype, removable_archetype_dc, __LINE__>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::remove(std::forward<decltype(policy)>(policy), view, search_value{searched});
        },
        // remove() returns the tail holding the removed elements, and the value occurs exactly once.
        [](auto&&, auto res) { return std::ranges::size(res) == 1; }, "remove");

    // nocopy_search_value is neither copyable nor movable: the host implementations must refer to
    // the value passed by the user instead of storing a copy of it.
    //
    // A device policy has to copy the value into the kernel, so the hetero runs cannot use that very
    // type and take its device copyable counterpart instead, which is still neither default
    // constructible nor ordered. The element archetype names the matching value type as
    // nocopy_value_type, so one generic lambda serves both sides.
    run_algo_all_policies<searchable_archetype, searchable_archetype_dc, __LINE__>(
        [](auto&& policy, auto&& view) {
            using elem_t = std::ranges::range_value_t<std::remove_cvref_t<decltype(view)>>;
            return dpl_ranges::find(std::forward<decltype(policy)>(policy), view,
                                    typename elem_t::nocopy_value_type{searched});
        },
        [](auto&& view, auto res) { return res == std::ranges::begin(view) + searched; }, "find, noncopyable value");

    run_algo_all_policies<searchable_archetype, searchable_archetype_dc, __LINE__>(
        [](auto&& policy, auto&& view) {
            using elem_t = std::ranges::range_value_t<std::remove_cvref_t<decltype(view)>>;
            return dpl_ranges::find_last(std::forward<decltype(policy)>(policy), view,
                                         typename elem_t::nocopy_value_type{searched});
        },
        [](auto&& view, auto res) { return std::ranges::begin(res) == std::ranges::begin(view) + searched; },
        "find_last, noncopyable value");

    // count() must refer to the value instead of storing a copy of it: the requires-clause never
    // asks for a copyable value type.
    run_algo_all_policies<searchable_archetype, searchable_archetype_dc, __LINE__>(
        [](auto&& policy, auto&& view) {
            using elem_t = std::ranges::range_value_t<std::remove_cvref_t<decltype(view)>>;
            return dpl_ranges::count(std::forward<decltype(policy)>(policy), view,
                                     typename elem_t::nocopy_value_type{searched});
        },
        [](auto&&, auto res) { return res == 1; }, "count, noncopyable value");

    run_algo_all_policies<searchable_archetype, searchable_archetype_dc, __LINE__>(
        [](auto&& policy, auto&& view) {
            using elem_t = std::ranges::range_value_t<std::remove_cvref_t<decltype(view)>>;
            return dpl_ranges::contains(std::forward<decltype(policy)>(policy), view,
                                        typename elem_t::nocopy_value_type{searched});
        },
        [](auto&&, auto res) { return res; }, "contains, noncopyable value");

    // Same for remove(): the predicate it builds internally must hold a reference to the value for
    // the host policies.
    run_algo_all_policies<removable_archetype, removable_archetype_dc, __LINE__>(
        [](auto&& policy, auto&& view) {
            using elem_t = std::ranges::range_value_t<std::remove_cvref_t<decltype(view)>>;
            return dpl_ranges::remove(std::forward<decltype(policy)>(policy), view,
                                      typename elem_t::nocopy_value_type{searched});
        },
        // remove() returns the tail holding the removed elements, and the value occurs exactly once.
        [](auto&&, auto res) { return std::ranges::size(res) == 1; }, "remove, noncopyable value");

    //----------------------------------------------------------------------------------------------
    // Callables taking their arguments by non-const reference: the value based algorithms with a
    // projection taking the element by non-const reference.
    //----------------------------------------------------------------------------------------------
    run_algo_all_policies<searchable_archetype, searchable_archetype_dc, __LINE__>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::find(std::forward<decltype(policy)>(policy), view, search_value{searched},
                                    search_proj_mut{});
        },
        [](auto&& view, auto res) { return res == std::ranges::begin(view) + searched; },
        "find, non-const projection");

    run_algo_all_policies<searchable_archetype, searchable_archetype_dc, __LINE__>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::find_last(std::forward<decltype(policy)>(policy), view, search_value{searched},
                                         search_proj_mut{});
        },
        [](auto&& view, auto res) { return std::ranges::begin(res) == std::ranges::begin(view) + searched; },
        "find_last, non-const projection");

    run_algo_all_policies<searchable_archetype, searchable_archetype_dc, __LINE__>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::count(std::forward<decltype(policy)>(policy), view, search_value{searched},
                                     search_proj_mut{});
        },
        [](auto&&, auto res) { return res == 1; }, "count, non-const projection");

    run_algo_all_policies<searchable_archetype, searchable_archetype_dc, __LINE__>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::contains(std::forward<decltype(policy)>(policy), view, search_value{searched},
                                        search_proj_mut{});
        },
        [](auto&&, auto res) { return res; }, "contains, non-const projection");

    run_algo_all_policies<removable_archetype, removable_archetype_dc, __LINE__>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::remove(std::forward<decltype(policy)>(policy), view, search_value{searched},
                                      search_proj_mut{});
        },
        // remove() returns the tail holding the removed elements, and the value 3 occurs exactly once.
        [](auto&&, auto res) { return std::ranges::size(res) == 1; }, "remove, non-const projection");

#endif //_ENABLE_STD_RANGES_TESTING

    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
