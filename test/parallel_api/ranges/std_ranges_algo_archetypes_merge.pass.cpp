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

    // The merge archetype family: the algorithms which are constrained by std::mergeable, i.e. which
    // read two sorted inputs and write into an output range of their own. Covers merge, set_union,
    // set_difference, set_intersection and set_symmetric_difference, first with const comparators and
    // then with comparators taking their arguments by non-const reference.

    // Both inputs hold the very same sorted sequence 0, 1, 2, ...
    run_algo2_all_policies<merge_in_archetype, merge_in_archetype, merge_in_archetype_dc, merge_in_archetype_dc,
                           __LINE__>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            using elem_t = std::ranges::range_value_t<std::remove_cvref_t<decltype(view1)>>;
            auto out_storage = make_out_storage<typename elem_t::out_type>(policy, 2 * archetype_test_size);
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
    //  - set_algorithms_utils.h:127,133,206,250,259 / memory_impl.h:96,111 -
    //    __uninitialized_copy_or_discard default constructs and copy constructs the output element
    //    type;
    //  - utils.h:1124 - the device path does the same through __lazy_ctor_storage::__setup, which
    //    placement news the output element and takes its address as well; it is reached from
    //    parallel_backend_sycl_reduce_then_scan.h:67,571,1049 for every set operation.
    // Fixing this means assigning through the output iterator instead of constructing in place.
    run_algo2_all_policies<merge_in_archetype, merge_in_archetype, merge_in_archetype_dc, merge_in_archetype_dc,
                           __LINE__, !_TEST_CPP20_RANGES_BROKEN_REQUIRES_SET_UNION_HOST,
                           !_TEST_CPP20_RANGES_BROKEN_REQUIRES_SET_UNION_HETERO>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            using elem_t = std::ranges::range_value_t<std::remove_cvref_t<decltype(view1)>>;
            auto out_storage = make_out_storage<typename elem_t::out_type>(policy, 2 * archetype_test_size);
            auto out_view = out_storage.view();
            auto res =
                dpl_ranges::set_union(std::forward<decltype(policy)>(policy), view1, view2, out_view, merge_comp{});
            // The two inputs hold the very same sequence, so the union is that sequence itself.
            return std::ranges::begin(out_view)[7].val == 7 &&
                   (std::size_t)(res.out - std::ranges::begin(out_view)) == archetype_test_size;
        },
        [](auto&&, auto&&, auto res) { return res; }, "set_union");

    run_algo2_all_policies<merge_in_archetype, merge_in_archetype, merge_in_archetype_dc, merge_in_archetype_dc,
                           __LINE__, !_TEST_CPP20_RANGES_BROKEN_REQUIRES_SET_DIFFERENCE_HOST,
                           !_TEST_CPP20_RANGES_BROKEN_REQUIRES_SET_DIFFERENCE_HETERO>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            using elem_t = std::ranges::range_value_t<std::remove_cvref_t<decltype(view1)>>;
            auto out_storage = make_out_storage<typename elem_t::out_type>(policy, 2 * archetype_test_size);
            auto out_view = out_storage.view();
            auto res = dpl_ranges::set_difference(std::forward<decltype(policy)>(policy), view1, view2, out_view,
                                                  merge_comp{});
            // The two inputs are equal, so the difference is empty.
            return res.out == std::ranges::begin(out_view);
        },
        [](auto&&, auto&&, auto res) { return res; }, "set_difference");

    // set_intersection and set_symmetric_difference construct the output element the very same way,
    // see the note above set_union.
    run_algo2_all_policies<merge_in_archetype, merge_in_archetype, merge_in_archetype_dc, merge_in_archetype_dc,
                           __LINE__, !_TEST_CPP20_RANGES_BROKEN_REQUIRES_SET_INTERSECTION_HOST,
                           !_TEST_CPP20_RANGES_BROKEN_REQUIRES_SET_INTERSECTION_HETERO>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            using elem_t = std::ranges::range_value_t<std::remove_cvref_t<decltype(view1)>>;
            auto out_storage = make_out_storage<typename elem_t::out_type>(policy, 2 * archetype_test_size);
            auto out_view = out_storage.view();
            auto res = dpl_ranges::set_intersection(std::forward<decltype(policy)>(policy), view1, view2, out_view,
                                                    merge_comp{});
            // The two inputs hold the very same sequence, so the intersection is that sequence itself.
            return std::ranges::begin(out_view)[7].val == 7 &&
                   (std::size_t)(res.out - std::ranges::begin(out_view)) == archetype_test_size;
        },
        [](auto&&, auto&&, auto res) { return res; }, "set_intersection");

    run_algo2_all_policies<merge_in_archetype, merge_in_archetype, merge_in_archetype_dc, merge_in_archetype_dc,
                           __LINE__, !_TEST_CPP20_RANGES_BROKEN_REQUIRES_SET_SYMMETRIC_DIFFERENCE_HOST,
                           !_TEST_CPP20_RANGES_BROKEN_REQUIRES_SET_SYMMETRIC_DIFFERENCE_HETERO>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            using elem_t = std::ranges::range_value_t<std::remove_cvref_t<decltype(view1)>>;
            auto out_storage = make_out_storage<typename elem_t::out_type>(policy, 2 * archetype_test_size);
            auto out_view = out_storage.view();
            auto res = dpl_ranges::set_symmetric_difference(std::forward<decltype(policy)>(policy), view1, view2,
                                                            out_view, merge_comp{});
            // The two inputs are equal, so the symmetric difference is empty.
            return res.out == std::ranges::begin(out_view);
        },
        [](auto&&, auto&&, auto res) { return res; }, "set_symmetric_difference");

    //----------------------------------------------------------------------------------------------
    // The same algorithms with callables taking their arguments by non-const reference.
    //----------------------------------------------------------------------------------------------
    // Both inputs hold the very same sorted sequence 0, 1, 2, ...
    run_algo2_all_policies<merge_in_archetype, merge_in_archetype, merge_in_archetype_dc, merge_in_archetype_dc,
                           __LINE__>(
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

    // The set operations, whose comparator is constrained exactly like the one of merge. They are
    // guarded by the very same macros as the const comparator cases above: the implementation
    // constructs the output element instead of assigning to it, which std::mergeable never asks for,
    // and that breaks the call before the comparator is ever reached.
    run_algo2_all_policies<merge_in_archetype, merge_in_archetype, merge_in_archetype_dc, merge_in_archetype_dc,
                           __LINE__, !_TEST_CPP20_RANGES_BROKEN_REQUIRES_SET_UNION_HOST,
                           !_TEST_CPP20_RANGES_BROKEN_REQUIRES_SET_UNION_HETERO>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            using elem_t = std::ranges::range_value_t<std::remove_cvref_t<decltype(view1)>>;
            auto out_storage = make_out_storage<typename elem_t::out_type>(policy, 2 * archetype_test_size);
            auto out_view = out_storage.view();
            auto res = dpl_ranges::set_union(std::forward<decltype(policy)>(policy), view1, view2, out_view,
                                             merge_comp_mut{});
            // The two inputs hold the very same sequence, so the union is that sequence itself.
            return std::ranges::begin(out_view)[7].val == 7 &&
                   (std::size_t)(res.out - std::ranges::begin(out_view)) == archetype_test_size;
        },
        [](auto&&, auto&&, auto res) { return res; }, "set_union, non-const comparator");

    run_algo2_all_policies<merge_in_archetype, merge_in_archetype, merge_in_archetype_dc, merge_in_archetype_dc,
                           __LINE__, !_TEST_CPP20_RANGES_BROKEN_REQUIRES_SET_DIFFERENCE_HOST,
                           !_TEST_CPP20_RANGES_BROKEN_REQUIRES_SET_DIFFERENCE_HETERO>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            using elem_t = std::ranges::range_value_t<std::remove_cvref_t<decltype(view1)>>;
            auto out_storage = make_out_storage<typename elem_t::out_type>(policy, 2 * archetype_test_size);
            auto out_view = out_storage.view();
            auto res = dpl_ranges::set_difference(std::forward<decltype(policy)>(policy), view1, view2, out_view,
                                                 merge_comp_mut{});
            // The two inputs are equal, so the difference is empty.
            return res.out == std::ranges::begin(out_view);
        },
        [](auto&&, auto&&, auto res) { return res; }, "set_difference, non-const comparator");

#endif //_ENABLE_STD_RANGES_TESTING

    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
