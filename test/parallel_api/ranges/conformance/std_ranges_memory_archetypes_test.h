// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) UXL Foundation Contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _STD_RANGES_MEMORY_ARCHETYPES_TEST_H
#define _STD_RANGES_MEMORY_ARCHETYPES_TEST_H

#if _ENABLE_STD_RANGES_TESTING

#include "../std_ranges_memory_test.h"
#include "std_ranges_archetypes.h"

#include <cstddef>
#include <cstring>
#include <memory>
#include <ranges>
#include <string>
#include <utility>

namespace test_std_ranges
{

template <typename Elem, typename Alloc, typename Policy, typename Algo, typename Checker>
void
run_over_archetype_view(Alloc& alloc, Policy&& policy, Algo algo, Checker checker, const char* algo_name)
{
    const std::size_t n = medium_size;
    Elem* data = alloc.allocate(n);
    std::memset(reinterpret_cast<void*>(data), -1, n * sizeof(Elem));

    archetypes::archetype_view<Elem> view(data, n);

    auto res = algo(std::forward<Policy>(policy), view);

    EXPECT_TRUE(res == view.begin() + n, (std::string("wrong return value from ") + algo_name +
                                          " over archetype_view").c_str());
    EXPECT_TRUE(std::ranges::all_of(view, checker), (std::string("wrong effect from ") + algo_name +
                                                     " over archetype_view").c_str());

    alloc.deallocate(data, n);
}

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
