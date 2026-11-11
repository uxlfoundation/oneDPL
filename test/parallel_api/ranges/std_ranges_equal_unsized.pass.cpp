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

#include "std_ranges_test.h"

#include <type_traits> // for std::is_invocable_v
#include <vector>

#if _ENABLE_STD_RANGES_TESTING
using infinite_t = std::ranges::iota_view<int, std::unreachable_sentinel_t>;

static_assert(std::ranges::random_access_range<infinite_t> && !std::ranges::sized_range<infinite_t>);

// At least one range must be sized
using policy_t = decltype(oneapi::dpl::execution::seq);
static_assert(std::is_invocable_v<decltype(oneapi::dpl::ranges::equal), policy_t, std::vector<int>&, infinite_t>);
static_assert(std::is_invocable_v<decltype(oneapi::dpl::ranges::equal), policy_t, infinite_t, std::vector<int>&>);
static_assert(!std::is_invocable_v<decltype(oneapi::dpl::ranges::equal), policy_t, infinite_t, infinite_t>);

template <typename Policy, typename TView1, typename TView2>
void
test_impl(Policy&& exec, TView1&& view1, TView2&& view2, bool expected)
{
    namespace dpl_ranges = oneapi::dpl::ranges;
    const char* err_msg = "Wrong effect algo equal with an infinite range.";

    // An infinite range is compared up to the size of the sized one
    EXPECT_TRUE(dpl_ranges::equal(exec, view1, view2) == expected, err_msg);
    EXPECT_TRUE(dpl_ranges::equal(std::forward<Policy>(exec), view2, view1) == expected, err_msg);
}

template <typename Policy>
void
test_host(Policy&& exec)
{
    std::vector<int> sized(test_std_ranges::medium_size);
    for (int i = 0; i < (int)sized.size(); ++i)
        sized[i] = i;

    // Equal elements
    test_impl(exec, sized, infinite_t(0), true);
    // Empty sized range
    test_impl(exec, std::ranges::subrange(sized.begin(), sized.begin()), infinite_t(0), true);
    // Different elements
    test_impl(exec, sized, infinite_t(1), false);

    std::vector<int> other = sized;
    other[other.size() / 2] = -1;
    test_impl(exec, other, infinite_t(0), false);
    other.back() = -1;
    other[other.size() / 2] = sized[other.size() / 2];
    test_impl(exec, other, infinite_t(0), false);
}
#endif // _ENABLE_STD_RANGES_TESTING

std::int32_t
main()
{
#if _ENABLE_STD_RANGES_TESTING
    test_host(oneapi::dpl::execution::seq);
    test_host(oneapi::dpl::execution::unseq);
    test_host(oneapi::dpl::execution::par);
    test_host(oneapi::dpl::execution::par_unseq);

#if TEST_DPCPP_BACKEND_PRESENT
    std::ranges::iota_view sized(0, test_std_ranges::medium_size);

    auto policy = TestUtils::get_dpcpp_test_policy();
    test_impl(policy, sized, infinite_t(0), true);
    test_impl(policy, sized, infinite_t(1), false);

#if TEST_CHECK_COMPILATION_WITH_DIFF_POLICY_VAL_CATEGORY
    TestUtils::check_compilation(policy, [&](auto&& policy) {
        test_impl(std::forward<decltype(policy)>(policy), sized, infinite_t(0), true);
    });
#endif
#endif // TEST_DPCPP_BACKEND_PRESENT
#endif // _ENABLE_STD_RANGES_TESTING

    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
