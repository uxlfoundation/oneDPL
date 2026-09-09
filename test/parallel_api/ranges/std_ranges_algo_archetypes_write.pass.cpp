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

    // None of the archetypes below is device copyable, so the host policies are the only ones the
    // constraints of these algorithms allow.
    run_algo_host_policies<writable_archetype>(
        [](auto&& policy, auto&& view) {
            using __elem = std::ranges::range_value_t<std::remove_cvref_t<decltype(view)>>;
            return dpl_ranges::fill(std::forward<decltype(policy)>(policy), view, typename __elem::value_arg{42});
        },
        [](auto&& view, auto) {
            return std::ranges::begin(view)[0].val == 42 &&
                   std::ranges::begin(view)[std::ranges::size(view) - 1].val == 42;
        },
        "fill");

#if TEST_DPCPP_BACKEND_PRESENT
    // None of the archetypes below is device copyable, so the host policies are the only ones the
    // constraints of these algorithms allow.
    run_algo_hetero_policies<writable_archetype_dc, 0>(
        [](auto&& policy, auto&& view) {
            using __elem = std::ranges::range_value_t<std::remove_cvref_t<decltype(view)>>;
            return dpl_ranges::fill(std::forward<decltype(policy)>(policy), view, typename __elem::value_arg{42});
        },
        [](auto&& view, auto) {
            return std::ranges::begin(view)[0].val == 42 &&
                   std::ranges::begin(view)[std::ranges::size(view) - 1].val == 42;
        },
        "fill");
#endif // TEST_DPCPP_BACKEND_PRESENT

    run_algo2_host_policies<copy_in_archetype, copy_out_archetype>(
        [](auto&& policy, auto&& in_view, auto&& out_view) {
            return dpl_ranges::copy(std::forward<decltype(policy)>(policy), in_view, out_view);
        },
        [](auto&& in_view, auto&& out_view, auto) {
            return std::ranges::begin(out_view)[7].val == std::ranges::begin(in_view)[7].val;
        },
        "copy");

#if TEST_DPCPP_BACKEND_PRESENT
    run_algo2_hetero_policies<copy_in_archetype_dc, copy_out_archetype_dc, 1>(
        [](auto&& policy, auto&& in_view, auto&& out_view) {
            return dpl_ranges::copy(std::forward<decltype(policy)>(policy), in_view, out_view);
        },
        [](auto&& in_view, auto&& out_view, auto) {
            return std::ranges::begin(out_view)[7].val == std::ranges::begin(in_view)[7].val;
        },
        "copy");
#endif // TEST_DPCPP_BACKEND_PRESENT

    run_algo2_host_policies<move_in_archetype, move_out_archetype>(
        [](auto&& policy, auto&& in_view, auto&& out_view) {
            return dpl_ranges::move(std::forward<decltype(policy)>(policy), in_view, out_view);
        },
        [](auto&&, auto&& out_view, auto) { return std::ranges::begin(out_view)[7].val == 7; }, "move");

#if TEST_DPCPP_BACKEND_PRESENT
    run_algo2_hetero_policies<move_in_archetype_dc, move_out_archetype_dc, 2>(
        [](auto&& policy, auto&& in_view, auto&& out_view) {
            return dpl_ranges::move(std::forward<decltype(policy)>(policy), in_view, out_view);
        },
        [](auto&&, auto&& out_view, auto) { return std::ranges::begin(out_view)[7].val == 7; }, "move");
#endif // TEST_DPCPP_BACKEND_PRESENT

    run_algo2_host_policies<swap_archetype, swap_archetype>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            return dpl_ranges::swap_ranges(std::forward<decltype(policy)>(policy), view1, view2);
        },
        [](auto&& view1, auto&& view2, auto) {
            return std::ranges::begin(view1)[7].val == 7 && std::ranges::begin(view2)[7].val == 7;
        },
        "swap_ranges");

#if TEST_DPCPP_BACKEND_PRESENT
    run_algo2_hetero_policies<swap_archetype_dc, swap_archetype_dc, 3>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            return dpl_ranges::swap_ranges(std::forward<decltype(policy)>(policy), view1, view2);
        },
        [](auto&& view1, auto&& view2, auto) {
            return std::ranges::begin(view1)[7].val == 7 && std::ranges::begin(view2)[7].val == 7;
        },
        "swap_ranges");
#endif // TEST_DPCPP_BACKEND_PRESENT

    run_algo2_host_policies<transform_in_archetype, transform_out_archetype>(
        [](auto&& policy, auto&& in_view, auto&& out_view) {
            return dpl_ranges::transform(std::forward<decltype(policy)>(policy), in_view, out_view,
                                         transform_unary_op{});
        },
        [](auto&&, auto&& out_view, auto) { return std::ranges::begin(out_view)[7].val == 14; }, "transform");

#if TEST_DPCPP_BACKEND_PRESENT
    run_algo2_hetero_policies<transform_in_archetype_dc, transform_out_archetype_dc, 4>(
        [](auto&& policy, auto&& in_view, auto&& out_view) {
            return dpl_ranges::transform(std::forward<decltype(policy)>(policy), in_view, out_view,
                                         transform_unary_op{});
        },
        [](auto&&, auto&& out_view, auto) { return std::ranges::begin(out_view)[7].val == 14; }, "transform");
#endif // TEST_DPCPP_BACKEND_PRESENT

    // The same overload with a non-identity projection: the functor is invoked with the projected
    // value, which is neither the element nor the output element type.
    run_algo2_host_policies<transform_in_archetype, transform_out_archetype>(
        [](auto&& policy, auto&& in_view, auto&& out_view) {
            return dpl_ranges::transform(std::forward<decltype(policy)>(policy), in_view, out_view,
                                         transform_projected_unary_op{}, transform_proj{});
        },
        [](auto&&, auto&& out_view, auto) { return std::ranges::begin(out_view)[7].val == 16; },
        "transform, projection");

#if TEST_DPCPP_BACKEND_PRESENT
    run_algo2_hetero_policies<transform_in_archetype_dc, transform_out_archetype_dc, 5>(
        [](auto&& policy, auto&& in_view, auto&& out_view) {
            return dpl_ranges::transform(std::forward<decltype(policy)>(policy), in_view, out_view,
                                         transform_projected_unary_op{}, transform_proj{});
        },
        [](auto&&, auto&& out_view, auto) { return std::ranges::begin(out_view)[7].val == 16; },
        "transform, projection");
#endif // TEST_DPCPP_BACKEND_PRESENT

    // The binary overload takes two input ranges, so the output range is allocated inside the call
    // and the check is done there as well.
    run_algo2_host_policies<transform_in_archetype, transform_in_archetype>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            archetype_storage<transform_out_archetype, std::allocator<transform_out_archetype>> out_storage(
                std::allocator<transform_out_archetype>{}, archetype_test_size, [](std::size_t) { return 0; });
            auto out_view = out_storage.view();
            auto res = dpl_ranges::transform(std::forward<decltype(policy)>(policy), view1, view2, out_view,
                                             transform_binary_op{});
            return std::ranges::begin(out_view)[7].val == 14 && res.out == std::ranges::end(out_view);
        },
        [](auto&&, auto&&, auto res) { return res; }, "transform, binary");

#if TEST_DPCPP_BACKEND_PRESENT
    run_algo2_hetero_policies<transform_in_archetype_dc, transform_in_archetype_dc, 6>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            // The output range is written by a device kernel, so its storage has to be device
            // accessible: host memory from std::allocator would be dereferenced on the device.
            sycl::usm_allocator<transform_out_archetype_dc, sycl::usm::alloc::shared> out_alloc{policy.queue()};
            archetype_storage<transform_out_archetype_dc, decltype(out_alloc)> out_storage(
                out_alloc, archetype_test_size, [](std::size_t) { return 0; });
            auto out_view = out_storage.view();
            auto res = dpl_ranges::transform(std::forward<decltype(policy)>(policy), view1, view2, out_view,
                                             transform_binary_op{});
            return std::ranges::begin(out_view)[7].val == 14 && res.out == std::ranges::end(out_view);
        },
        [](auto&&, auto&&, auto res) { return res; }, "transform, binary");
#endif // TEST_DPCPP_BACKEND_PRESENT

    // The binary overload has a projection of its own for either input.
    run_algo2_host_policies<transform_in_archetype, transform_in_archetype>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            archetype_storage<transform_out_archetype, std::allocator<transform_out_archetype>> out_storage(
                std::allocator<transform_out_archetype>{}, archetype_test_size, [](std::size_t) { return 0; });
            auto out_view = out_storage.view();
            auto res = dpl_ranges::transform(std::forward<decltype(policy)>(policy), view1, view2, out_view,
                                             transform_projected_binary_op{}, transform_proj{}, transform_proj{});
            return std::ranges::begin(out_view)[7].val == 16 && res.out == std::ranges::end(out_view);
        },
        [](auto&&, auto&&, auto res) { return res; }, "transform, binary, projections");

#if TEST_DPCPP_BACKEND_PRESENT
    run_algo2_hetero_policies<transform_in_archetype_dc, transform_in_archetype_dc, 7>(
        [](auto&& policy, auto&& view1, auto&& view2) {
            sycl::usm_allocator<transform_out_archetype_dc, sycl::usm::alloc::shared> out_alloc{policy.queue()};
            archetype_storage<transform_out_archetype_dc, decltype(out_alloc)> out_storage(
                out_alloc, archetype_test_size, [](std::size_t) { return 0; });
            auto out_view = out_storage.view();
            auto res = dpl_ranges::transform(std::forward<decltype(policy)>(policy), view1, view2, out_view,
                                             transform_projected_binary_op{}, transform_proj{}, transform_proj{});
            return std::ranges::begin(out_view)[7].val == 16 && res.out == std::ranges::end(out_view);
        },
        [](auto&&, auto&&, auto res) { return res; }, "transform, binary, projections");
#endif // TEST_DPCPP_BACKEND_PRESENT
#endif //_ENABLE_STD_RANGES_TESTING

    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
