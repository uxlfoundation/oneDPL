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

#ifndef _STD_RANGES_MEMORY_ARCHETYPES_TEST_H
#define _STD_RANGES_MEMORY_ARCHETYPES_TEST_H

#if _ENABLE_STD_RANGES_TESTING

// The harness of the memory algorithms over the archetypes. It is kept apart from
// std_ranges_memory_test.h so that the pre-existing memory tests do not have to parse the archetypes,
// and apart from std_ranges_algo_archetypes_test.h because the memory algorithms work over raw
// uninitialized storage: the elements are not constructed yet, so archetype_storage, which constructs
// every element in its constructor, cannot be used here.
#include "std_ranges_memory_test.h"
#include "std_ranges_archetypes.h"

#include <cstddef>
#include <cstring>
#include <memory>
#include <ranges>
#include <string>
#include <utility>

namespace test_std_ranges
{

// Runs a one-range algorithm over archetype_view, which is random access and sized but neither
// contiguous nor common, so the implementation cannot fall back to raw pointer arithmetic.
template <typename Elem, typename Alloc, typename Policy, typename Algo, typename Checker>
void
run_over_archetype_view(Alloc& alloc, Policy&& policy, Algo algo, Checker checker, const char* algo_name)
{
    const std::size_t n = medium_size;
    Elem* data = alloc.allocate(n);
    std::memset(reinterpret_cast<void*>(data), -1, n * sizeof(Elem)); // -1 means no initialization

    archetypes::archetype_view<Elem> view(data, n);

    auto res = algo(std::forward<Policy>(policy), view);

    EXPECT_TRUE(res == view.begin() + n, (std::string("wrong return value from ") + algo_name +
                                          " over archetype_view").c_str());
    EXPECT_TRUE(std::ranges::all_of(view, checker), (std::string("wrong effect from ") + algo_name +
                                                     " over archetype_view").c_str());

    alloc.deallocate(data, n);
}

// call_id makes the SYCL kernel name of the device call unique within a translation unit, see
// test_memory_algo in std_ranges_memory_test.h.
template <typename Elem, int call_id, typename Algo, typename Checker>
void
run_archetype_view_all_policies(Algo algo, Checker checker, const char* algo_name)
{
    std::allocator<Elem> alloc;
    run_over_archetype_view<Elem>(alloc, oneapi::dpl::execution::seq, algo, checker, algo_name);
    run_over_archetype_view<Elem>(alloc, oneapi::dpl::execution::unseq, algo, checker, algo_name);
    run_over_archetype_view<Elem>(alloc, oneapi::dpl::execution::par, algo, checker, algo_name);
    run_over_archetype_view<Elem>(alloc, oneapi::dpl::execution::par_unseq, algo, checker, algo_name);

#if TEST_DPCPP_BACKEND_PRESENT
    auto policy = TestUtils::get_dpcpp_test_policy<call_id>();
    sycl::usm_allocator<Elem, sycl::usm::alloc::shared> q_alloc{policy.queue()};
    run_over_archetype_view<Elem>(q_alloc, policy, algo, checker, algo_name);
#endif //TEST_DPCPP_BACKEND_PRESENT
}

} //namespace test_std_ranges

#endif // _ENABLE_STD_RANGES_TESTING
#endif // _STD_RANGES_MEMORY_ARCHETYPES_TEST_H
