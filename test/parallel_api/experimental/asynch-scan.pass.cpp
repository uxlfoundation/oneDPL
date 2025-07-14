// -*- C++ -*-
//===-- async-scan.pass.cpp ----------------------------------------------------===//
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

#include "support/test_config.h"

#if TEST_DPCPP_BACKEND_PRESENT
#   include "oneapi/dpl/async"
#endif // TEST_DPCPP_BACKEND_PRESENT
#include "oneapi/dpl/execution"
#include "oneapi/dpl/iterator"

#include "support/utils.h"

#include <iostream>
#include <iomanip>
#include <numeric>

#if TEST_DPCPP_BACKEND_PRESENT

class Copy;

template <std::size_t idx>
class Scan;

struct MultiplyBy10FO
{
    template <typename T>
    T
    operator()(T x) const
    {
        return x * 10;
    }
};

template <typename Policy>
void
test_with_buffers(Policy&& exec)
{
    const int n = 100;

    sycl::buffer<int> x{n};
    sycl::buffer<int> y{n};

    auto input = oneapi::dpl::counting_iterator<int>(0);

    dpl::experimental::copy_async(CLONE_TEST_POLICY(exec), input, input+n, dpl::begin(x)).wait();
    const auto expected1 = ((n-1)*n)/2;
    const auto expected2 = expected1-n+1;

    // transform inclusive (2 overloads)
    auto alpha = dpl::experimental::transform_inclusive_scan_async(
        CLONE_TEST_POLICY_NAME(exec, Scan<1>),
        dpl::begin(x), dpl::end(x), dpl::begin(y), std::plus<int>(), MultiplyBy10FO{});
    auto result1 = alpha.get().get_buffer().get_host_access(sycl::read_only)[n-1];
    EXPECT_TRUE(result1 == (expected1 * 10), "wrong effect from async scan test (Ia) with sycl buffer");

    auto fut1b = dpl::experimental::transform_inclusive_scan_async(
        CLONE_TEST_POLICY_NAME(exec, Scan<2>),
        dpl::begin(x), dpl::end(x), dpl::begin(y), std::plus<int>(), MultiplyBy10FO{}, 1);
    auto result1b = fut1b.get().get_buffer().get_host_access(sycl::read_only)[n-1];
    EXPECT_TRUE(result1b == (expected1 * 10 + 1), "wrong effect from async scan test (Ib) with sycl buffer");

    // transform exclusive
    auto beta = dpl::experimental::transform_exclusive_scan_async(
        CLONE_TEST_POLICY_NAME(exec, Scan<3>),
        dpl::begin(x), dpl::end(x), dpl::begin(y), 0, std::plus<int>(), MultiplyBy10FO{});
    auto result2 = beta.get().get_buffer().get_host_access(sycl::read_only)[n-1];
    EXPECT_TRUE(result2 == expected2 * 10, "wrong effect from async scan test (II) with sycl buffer");

    // inclusive (3 overloads)
    auto gamma = dpl::experimental::inclusive_scan_async(
        CLONE_TEST_POLICY_NAME(exec, Scan<4>),
        dpl::begin(x), dpl::end(x), dpl::begin(y));
    auto result3 = gamma.get().get_buffer().get_host_access(sycl::read_only)[n-1];
    EXPECT_TRUE(result3 == expected1, "wrong effect from async scan test (IIIa) with sycl buffer");

    auto fut3b = dpl::experimental::inclusive_scan_async(
        CLONE_TEST_POLICY_NAME(exec, Scan<5>),
        dpl::begin(x), dpl::end(x), dpl::begin(y), std::plus<int>(), gamma);
    auto result3b = fut3b.get().get_buffer().get_host_access(sycl::read_only)[n-1];
    EXPECT_TRUE(result3b == expected1, "wrong effect from async scan test (IIIb) with sycl buffer");

    auto fut3c = dpl::experimental::inclusive_scan_async(
        CLONE_TEST_POLICY_NAME(exec, Scan<6>),
        dpl::begin(x), dpl::end(x), dpl::begin(y), std::plus<int>(), 1, fut3b);
    auto result3c = fut3c.get().get_buffer().get_host_access(sycl::read_only)[n-1];
    EXPECT_TRUE(result3c == (expected1 + 1), "wrong effect from async scan test (IIIc) with sycl buffer");

    // exclusive (2 overloads)
    auto delta = dpl::experimental::exclusive_scan_async(
        CLONE_TEST_POLICY_NAME(exec, Scan<7>),
        dpl::begin(x), dpl::end(x), dpl::begin(y), 0);
    auto result4 = delta.get().get_buffer().get_host_access(sycl::read_only)[n-1];
    EXPECT_TRUE(result4 == expected2, "wrong effect from async scan test (IV) with sycl buffer");

    auto fut4b = dpl::experimental::exclusive_scan_async(
        CLONE_TEST_POLICY_NAME(exec, Scan<8>),
        dpl::begin(x), dpl::end(x), dpl::begin(y), 1, std::plus<int>(), delta);
    auto result4b = fut4b.get().get_buffer().get_host_access(sycl::read_only)[n-1];
    EXPECT_TRUE(result4b == (expected2 + 1), "wrong effect from async scan test (IV) with sycl buffer");

    oneapi::dpl::experimental::wait_for_all(alpha,beta,gamma,delta);
}

#endif // TEST_DPCPP_BACKEND_PRESENT

int main()
{
#if TEST_DPCPP_BACKEND_PRESENT

    auto policy = TestUtils::get_dpcpp_test_policy();
    test_with_buffers(policy);

#if TEST_CHECK_COMPILATION_WITH_DIFF_POLICY_VAL_CATEGORY
    TestUtils::check_compilation(policy, [](auto&& policy) { test_with_buffers(std::forward<decltype(policy)>(policy)); });
#endif
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
