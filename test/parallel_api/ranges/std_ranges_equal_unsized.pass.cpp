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

#include <numeric> // for std::iota
#include <vector>

#if _ENABLE_STD_RANGES_TESTING
// A sentinel that is not a sized_sentinel_for its iterator, which makes a finite random access range unsized
template <typename It>
struct unsized_sentinel
{
    It end{};

    friend bool
    operator==(const It& it, const unsized_sentinel& s)
    {
        return it == s.end;
    }
};

template <typename It>
auto
make_unsized(It first, It last)
{
    return std::ranges::subrange(first, unsized_sentinel<It>{last});
}

template <typename Policy, typename TView1, typename TView2>
void
test_impl(Policy&& exec, TView1&& view1, TView2&& view2, bool expected)
{
    namespace dpl_ranges = oneapi::dpl::ranges;
    const char* err_msg = "Wrong effect algo equal with unsized ranges.";

    EXPECT_TRUE(dpl_ranges::equal(exec, view1, view2) == expected, err_msg);
    EXPECT_TRUE(dpl_ranges::equal(std::forward<Policy>(exec), view2, view1) == expected, err_msg);
}

template <typename Policy>
void
test_host(Policy&& exec)
{
    const int n = test_std_ranges::medium_size;
    std::vector<int> sized(n);
    std::iota(sized.begin(), sized.end(), 0);

    // Holds one element more than "sized", to build unsized ranges that are longer or shorter by one
    std::vector<int> data(n + 1);
    std::iota(data.begin(), data.end(), 0);

    static_assert(!std::ranges::sized_range<decltype(make_unsized(data.begin(), data.end()))>);
    static_assert(std::ranges::random_access_range<decltype(make_unsized(data.begin(), data.end()))>);

    // Infinite range
    test_impl(exec, sized, std::ranges::iota_view(0, std::unreachable_sentinel_t{}), false);
    // Finite unsized ranges: equal, shorter, longer
    test_impl(exec, sized, make_unsized(data.begin(), data.begin() + n), true);
    test_impl(exec, sized, make_unsized(data.begin(), data.begin() + n - 1), false);
    test_impl(exec, sized, make_unsized(data.begin(), data.end()), false);

    // Same length, one element differs
    std::vector<int> other = sized;
    other[n / 2] = -1;
    test_impl(exec, sized, make_unsized(other.begin(), other.end()), false);

    // Empty ranges
    test_impl(exec, std::ranges::subrange(sized.begin(), sized.begin()), make_unsized(data.begin(), data.begin()),
              true);
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
    const int n = test_std_ranges::medium_size;
    std::ranges::iota_view sized(0, n);
    std::ranges::iota_view infinite(0, std::unreachable_sentinel_t{});

    auto policy = TestUtils::get_dpcpp_test_policy();
    test_impl(policy, sized, infinite, false);
    test_impl(policy, sized, make_unsized(infinite.begin(), infinite.begin() + n), true);
    test_impl(policy, sized, make_unsized(infinite.begin(), infinite.begin() + n + 1), false);

#if TEST_CHECK_COMPILATION_WITH_DIFF_POLICY_VAL_CATEGORY
    TestUtils::check_compilation(policy, [&](auto&& policy) {
        test_impl(std::forward<decltype(policy)>(policy), sized, infinite, false);
    });
#endif
#endif // TEST_DPCPP_BACKEND_PRESENT
#endif // _ENABLE_STD_RANGES_TESTING

    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
