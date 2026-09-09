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

    // The write archetype families: an element which is only assignable, from an unrelated value type
    // (fill), from the element of another range (copy, move, swap_ranges) or from the result of a
    // functor (transform, with and without projections). Nothing here is copyable, movable or default
    // constructible, and the elements written from are of a different type than the ones written to.

    //----------------------------------------------------------------------------------------------
    // The writing algorithms; every callable takes its arguments by const reference.
    //----------------------------------------------------------------------------------------------
    run_algo_all_policies<writable_archetype, writable_archetype_dc, __LINE__>(
        [](auto&& policy, auto&& view) {
            using __elem = std::ranges::range_value_t<std::remove_cvref_t<decltype(view)>>;
            return dpl_ranges::fill(std::forward<decltype(policy)>(policy), view, typename __elem::value_arg{42});
        },
        [](auto&& view, auto) {
            return std::ranges::begin(view)[0].val == 42 &&
                   std::ranges::begin(view)[std::ranges::size(view) - 1].val == 42;
        },
        "fill");

    run_algo2_all_policies<copy_in_archetype, copy_out_archetype, copy_in_archetype_dc, copy_out_archetype_dc,
                           __LINE__>(
        [](auto&& policy, auto&& in_view, auto&& out_view) {
            return dpl_ranges::copy(std::forward<decltype(policy)>(policy), in_view, out_view);
        },
        [](auto&& in_view, auto&& out_view, auto) {
            return std::ranges::begin(out_view)[7].val == std::ranges::begin(in_view)[7].val;
        },
        "copy");

    run_algo2_all_policies<move_in_archetype, move_out_archetype, move_in_archetype_dc, move_out_archetype_dc,
                           __LINE__>(
        [](auto&& policy, auto&& in_view, auto&& out_view) {
            return dpl_ranges::move(std::forward<decltype(policy)>(policy), in_view, out_view);
        },
        [](auto&&, auto&& out_view, auto) { return std::ranges::begin(out_view)[7].val == 7; }, "move");

    run_algo2_all_policies<swap_archetype, swap_archetype, swap_archetype_dc, swap_archetype_dc, __LINE__>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            return dpl_ranges::swap_ranges(std::forward<decltype(policy)>(policy), view1, view2);
        },
        [](auto&& view1, auto&& view2, auto) {
            return std::ranges::begin(view1)[7].val == 7 && std::ranges::begin(view2)[7].val == 7;
        },
        "swap_ranges");

    run_algo2_all_policies<transform_in_archetype, transform_out_archetype, transform_in_archetype_dc,
                           transform_out_archetype_dc, __LINE__>(
        [](auto&& policy, auto&& in_view, auto&& out_view) {
            return dpl_ranges::transform(std::forward<decltype(policy)>(policy), in_view, out_view,
                                         transform_unary_op{});
        },
        [](auto&&, auto&& out_view, auto) { return std::ranges::begin(out_view)[7].val == 14; }, "transform");

    // The same overload with a non-identity projection: the functor is invoked with the projected
    // value, which is neither the element nor the output element type.
    run_algo2_all_policies<transform_in_archetype, transform_out_archetype, transform_in_archetype_dc,
                           transform_out_archetype_dc, __LINE__>(
        [](auto&& policy, auto&& in_view, auto&& out_view) {
            return dpl_ranges::transform(std::forward<decltype(policy)>(policy), in_view, out_view,
                                         transform_projected_unary_op{}, transform_proj{});
        },
        [](auto&&, auto&& out_view, auto) { return std::ranges::begin(out_view)[7].val == 16; },
        "transform, projection");

    // The binary overload takes two input ranges, so the output range is allocated inside the call
    // and the check is done there as well.
    run_algo2_all_policies<transform_in_archetype, transform_in_archetype, transform_in_archetype_dc,
                           transform_in_archetype_dc, __LINE__>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            using elem_t = std::ranges::range_value_t<std::remove_cvref_t<decltype(view1)>>;
            auto out_storage = make_out_storage<typename elem_t::out_type>(policy, archetype_test_size);
            auto out_view = out_storage.view();
            auto res = dpl_ranges::transform(std::forward<decltype(policy)>(policy), view1, view2, out_view,
                                             transform_binary_op{});
            return std::ranges::begin(out_view)[7].val == 14 && res.out == std::ranges::end(out_view);
        },
        [](auto&&, auto&&, auto res) { return res; }, "transform, binary");

    // The binary overload has a projection of its own for either input.
    run_algo2_all_policies<transform_in_archetype, transform_in_archetype, transform_in_archetype_dc,
                           transform_in_archetype_dc, __LINE__>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            using elem_t = std::ranges::range_value_t<std::remove_cvref_t<decltype(view1)>>;
            auto out_storage = make_out_storage<typename elem_t::out_type>(policy, archetype_test_size);
            auto out_view = out_storage.view();
            auto res = dpl_ranges::transform(std::forward<decltype(policy)>(policy), view1, view2, out_view,
                                             transform_projected_binary_op{}, transform_proj{}, transform_proj{});
            return std::ranges::begin(out_view)[7].val == 16 && res.out == std::ranges::end(out_view);
        },
        [](auto&&, auto&&, auto res) { return res; }, "transform, binary, projections");

    //----------------------------------------------------------------------------------------------
    // The same algorithms with callables taking their arguments by non-const reference.
    //----------------------------------------------------------------------------------------------
    run_algo2_all_policies<transform_in_archetype, transform_out_archetype, transform_in_archetype_dc,
                           transform_out_archetype_dc, __LINE__>(
        [](auto&& policy, auto&& in_view, auto&& out_view) {
            return dpl_ranges::transform(std::forward<decltype(policy)>(policy), in_view, out_view,
                                         transform_unary_op_mut{});
        },
        [](auto&&, auto&& out_view, auto) { return std::ranges::begin(out_view)[7].val == 14; },
        "transform, non-const callable");

    // The projection is the one taking the element by non-const reference here: the functor is
    // invoked with the projected prvalue and cannot take it by non-const reference at all.
    run_algo2_all_policies<transform_in_archetype, transform_out_archetype, transform_in_archetype_dc,
                           transform_out_archetype_dc, __LINE__>(
        [](auto&& policy, auto&& in_view, auto&& out_view) {
            return dpl_ranges::transform(std::forward<decltype(policy)>(policy), in_view, out_view,
                                         transform_projected_unary_op{}, transform_proj_mut{});
        },
        [](auto&&, auto&& out_view, auto) { return std::ranges::begin(out_view)[7].val == 16; },
        "transform, non-const projection");

    // The binary overload with a functor taking both input elements by non-const reference. It takes
    // two input ranges, so the output range is allocated inside the call and checked there as well.
    run_algo2_all_policies<transform_in_archetype, transform_in_archetype, transform_in_archetype_dc,
                           transform_in_archetype_dc, __LINE__>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            using elem_t = std::ranges::range_value_t<std::remove_cvref_t<decltype(view1)>>;
            auto out_storage = make_out_storage<typename elem_t::out_type>(policy, archetype_test_size);
            auto out_view = out_storage.view();
            auto res = dpl_ranges::transform(std::forward<decltype(policy)>(policy), view1, view2, out_view,
                                             transform_binary_op_mut{});
            return std::ranges::begin(out_view)[7].val == 14 && res.out == std::ranges::end(out_view);
        },
        [](auto&&, auto&&, auto res) { return res; }, "transform, binary, non-const callable");

    // The binary overload with a non-const projection for either input.
    run_algo2_all_policies<transform_in_archetype, transform_in_archetype, transform_in_archetype_dc,
                           transform_in_archetype_dc, __LINE__>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            using elem_t = std::ranges::range_value_t<std::remove_cvref_t<decltype(view1)>>;
            auto out_storage = make_out_storage<typename elem_t::out_type>(policy, archetype_test_size);
            auto out_view = out_storage.view();
            auto res =
                dpl_ranges::transform(std::forward<decltype(policy)>(policy), view1, view2, out_view,
                                      transform_projected_binary_op{}, transform_proj_mut{}, transform_proj_mut{});
            return std::ranges::begin(out_view)[7].val == 16 && res.out == std::ranges::end(out_view);
        },
        [](auto&&, auto&&, auto res) { return res; }, "transform, binary, non-const projections");
#endif //_ENABLE_STD_RANGES_TESTING

    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
