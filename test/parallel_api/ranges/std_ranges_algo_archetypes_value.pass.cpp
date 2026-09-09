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

// The value based algorithms are constrained by
//   std::indirect_binary_predicate<std::ranges::equal_to, std::projected<iterator_t<_R>, _Proj>,
//                                  const _T*>
// only. In particular the value type is not required to be copyable, to be comparable with itself
// with anything but std::ranges::equal_to, or to be related to the element type in any other way,
// and the element type is not required to be comparable with itself either; the device copyable
// counterpart of the value satisfies the very same constraints and really is accepted by SYCL without
// an explicit sycl::is_device_copyable specialization. That is asserted on the archetypes themselves
// in std_ranges_archetypes.h; what the calls below add is the instantiation of the implementation,
// which is where an extra requirement shows up as a compile error.

#endif //_ENABLE_STD_RANGES_TESTING

int
main()
{
#if _ENABLE_STD_RANGES_TESTING
    using namespace test_std_ranges;
    using namespace test_std_ranges::archetypes;
    namespace dpl_ranges = oneapi::dpl::ranges;

    // The storage is filled with the values 0, 1, 2, ... so the value 3 is found exactly once.
    constexpr int searched = 3;

    // search_value is trivially copyable and thus device copyable, so it can be used with all the
    // policies including the device ones.
    run_algo_host_policies<searchable_archetype>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::find(std::forward<decltype(policy)>(policy), view, search_value{searched});
        },
        [](auto&& view, auto res) { return res == std::ranges::begin(view) + searched; }, "find");

#if TEST_DPCPP_BACKEND_PRESENT
    run_algo_hetero_policies<searchable_archetype_dc, 0>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::find(std::forward<decltype(policy)>(policy), view, search_value{searched});
        },
        [](auto&& view, auto res) { return res == std::ranges::begin(view) + searched; }, "find");
#endif // TEST_DPCPP_BACKEND_PRESENT

    run_algo_host_policies<searchable_archetype>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::find_last(std::forward<decltype(policy)>(policy), view, search_value{searched});
        },
        [](auto&& view, auto res) { return std::ranges::begin(res) == std::ranges::begin(view) + searched; },
        "find_last");

#if TEST_DPCPP_BACKEND_PRESENT
    run_algo_hetero_policies<searchable_archetype_dc, 1>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::find_last(std::forward<decltype(policy)>(policy), view, search_value{searched});
        },
        [](auto&& view, auto res) { return std::ranges::begin(res) == std::ranges::begin(view) + searched; },
        "find_last");
#endif

    run_algo_host_policies<searchable_archetype>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::count(std::forward<decltype(policy)>(policy), view, search_value{searched});
        },
        [](auto&&, auto res) { return res == 1; }, "count");

#if TEST_DPCPP_BACKEND_PRESENT
    run_algo_hetero_policies<searchable_archetype_dc, 2>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::count(std::forward<decltype(policy)>(policy), view, search_value{searched});
        },
        [](auto&&, auto res) { return res == 1; }, "count");
#endif // TEST_DPCPP_BACKEND_PRESENT

    run_algo_host_policies<searchable_archetype>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::contains(std::forward<decltype(policy)>(policy), view, search_value{searched});
        },
        [](auto&&, auto res) { return res; }, "contains");

#if TEST_DPCPP_BACKEND_PRESENT
    run_algo_hetero_policies<searchable_archetype_dc, 3>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::contains(std::forward<decltype(policy)>(policy), view, search_value{searched});
        },
        [](auto&&, auto res) { return res; }, "contains");
#endif // TEST_DPCPP_BACKEND_PRESENT

    // removable_archetype is movable but not device copyable, so remove() is checked on the host
    // policies only.
    run_algo_host_policies<removable_archetype>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::remove(std::forward<decltype(policy)>(policy), view, search_value{searched});
        },
        // remove() returns the tail holding the removed elements, and the value occurs exactly once.
        [](auto&&, auto res) { return std::ranges::size(res) == 1; }, "remove");

#if TEST_DPCPP_BACKEND_PRESENT
    // removable_archetype is movable but not device copyable, so remove() is checked on the host
    // policies only.
    run_algo_hetero_policies<removable_archetype_dc, 4>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::remove(std::forward<decltype(policy)>(policy), view, search_value{searched});
        },
        // remove() returns the tail holding the removed elements, and the value occurs exactly once.
        [](auto&&, auto res) { return std::ranges::size(res) == 1; }, "remove");
#endif // TEST_DPCPP_BACKEND_PRESENT

    // nocopy_search_value is neither copyable nor movable: the host implementations must refer to
    // the value passed by the user instead of storing a copy of it. It cannot be captured by a
    // device kernel, hence the host policies only.
    run_algo_host_policies<searchable_archetype>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::find(std::forward<decltype(policy)>(policy), view, nocopy_search_value{searched});
        },
        [](auto&& view, auto res) { return res == std::ranges::begin(view) + searched; }, "find, noncopyable value");

#if TEST_DPCPP_BACKEND_PRESENT
    // A device policy copies the value into the kernel, so the hetero runs use the device copyable
    // counterpart of the value: it is still neither default constructible nor ordered.
    run_algo_hetero_policies<searchable_archetype_dc, 5>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::find(std::forward<decltype(policy)>(policy), view, nocopy_search_value_dc{searched});
        },
        [](auto&& view, auto res) { return res == std::ranges::begin(view) + searched; }, "find, noncopyable value");
#endif // TEST_DPCPP_BACKEND_PRESENT

    run_algo_host_policies<searchable_archetype>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::find_last(std::forward<decltype(policy)>(policy), view, nocopy_search_value{searched});
        },
        [](auto&& view, auto res) { return std::ranges::begin(res) == std::ranges::begin(view) + searched; },
        "find_last, noncopyable value");

#if TEST_DPCPP_BACKEND_PRESENT
    run_algo_hetero_policies<searchable_archetype_dc, 6>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::find_last(std::forward<decltype(policy)>(policy), view,
                                         nocopy_search_value_dc{searched});
        },
        [](auto&& view, auto res) { return std::ranges::begin(res) == std::ranges::begin(view) + searched; },
        "find_last, noncopyable value");
#endif // TEST_DPCPP_BACKEND_PRESENT

    // count() must refer to the value instead of storing a copy of it: the requires-clause never
    // asks for a copyable value type.
    run_algo_host_policies<searchable_archetype>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::count(std::forward<decltype(policy)>(policy), view, nocopy_search_value{searched});
        },
        [](auto&&, auto res) { return res == 1; }, "count, noncopyable value");

#if TEST_DPCPP_BACKEND_PRESENT
    run_algo_hetero_policies<searchable_archetype_dc, 7>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::count(std::forward<decltype(policy)>(policy), view, nocopy_search_value_dc{searched});
        },
        [](auto&&, auto res) { return res == 1; }, "count, noncopyable value");
#endif // TEST_DPCPP_BACKEND_PRESENT
    
    run_algo_host_policies<searchable_archetype>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::contains(std::forward<decltype(policy)>(policy), view, nocopy_search_value{searched});
        },
        [](auto&&, auto res) { return res; }, "contains, noncopyable value");

#if TEST_DPCPP_BACKEND_PRESENT
    run_algo_hetero_policies<searchable_archetype_dc, 8>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::contains(std::forward<decltype(policy)>(policy), view, nocopy_search_value_dc{searched});
        },
        [](auto&&, auto res) { return res; }, "contains, noncopyable value");
#endif // TEST_DPCPP_BACKEND_PRESENT

    // Same for remove(): the predicate it builds internally must hold a reference to the value for
    // the host policies.
    run_algo_host_policies<removable_archetype>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::remove(std::forward<decltype(policy)>(policy), view, nocopy_search_value{searched});
        },
        // remove() returns the tail holding the removed elements, and the value occurs exactly once.
        [](auto&&, auto res) { return std::ranges::size(res) == 1; }, "remove, noncopyable value");

#if TEST_DPCPP_BACKEND_PRESENT
    run_algo_hetero_policies<removable_archetype_dc, 9>(
        [](auto&& policy, auto&& view) {
            return dpl_ranges::remove(std::forward<decltype(policy)>(policy), view, nocopy_search_value_dc{searched});
        },
        // remove() returns the tail holding the removed elements, and the value occurs exactly once.
        [](auto&&, auto res) { return std::ranges::size(res) == 1; }, "remove, noncopyable value");
#endif // TEST_DPCPP_BACKEND_PRESENT

#endif //_ENABLE_STD_RANGES_TESTING

    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
