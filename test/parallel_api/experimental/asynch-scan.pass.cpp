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
void
test_with_buffers()
{
    const int n = 100;
    {
        sycl::buffer<int> x{n};
        sycl::buffer<int> y{n};

        auto my_policy = TestUtils::make_device_policy<class Copy>(oneapi::dpl::execution::dpcpp_default);
        auto input = oneapi::dpl::counting_iterator<int>(0);

        dpl::experimental::copy_async(my_policy, input, input+n, dpl::begin(x)).wait();
        const auto expected1 = ((n-1)*n)/2;
        const auto expected2 = expected1-n+1;

        // transform inclusive (2 overloads)
        auto my_policy1 = TestUtils::make_device_policy<class Scan1>(my_policy);
        auto alpha = dpl::experimental::transform_inclusive_scan_async(my_policy1, dpl::begin(x), dpl::end(x), dpl::begin(y), std::plus<int>(), [](auto x) { return x * 10; });
        auto result1 = alpha.get().get_buffer().get_host_access(sycl::read_only)[n-1];
        EXPECT_TRUE(result1 == (expected1 * 10), "wrong effect from async scan test (Ia) with sycl buffer");

        auto my_policy2 = TestUtils::make_device_policy<class Scan2>(my_policy);
        auto fut1b = dpl::experimental::transform_inclusive_scan_async(my_policy2, dpl::begin(x), dpl::end(x), dpl::begin(y), std::plus<int>(), [](auto x) { return x * 10; }, 1);
        auto result1b = fut1b.get().get_buffer().get_host_access(sycl::read_only)[n-1];
        EXPECT_TRUE(result1b == (expected1 * 10 + 1), "wrong effect from async scan test (Ib) with sycl buffer");

        // transform exclusive
        auto my_policy3 = TestUtils::make_device_policy<class Scan3>(my_policy);
        auto beta = dpl::experimental::transform_exclusive_scan_async(my_policy3, dpl::begin(x), dpl::end(x), dpl::begin(y), 0, std::plus<int>(), [](auto x) { return x * 10; });
        auto result2 = beta.get().get_buffer().get_host_access(sycl::read_only)[n-1];
        EXPECT_TRUE(result2 == expected2 * 10, "wrong effect from async scan test (II) with sycl buffer");

        // inclusive (3 overloads)
        auto my_policy4 = TestUtils::make_device_policy<class Scan4>(my_policy);
        auto gamma = dpl::experimental::inclusive_scan_async(my_policy4, dpl::begin(x), dpl::end(x), dpl::begin(y));
        auto result3 = gamma.get().get_buffer().get_host_access(sycl::read_only)[n-1];
        EXPECT_TRUE(result3 == expected1, "wrong effect from async scan test (IIIa) with sycl buffer");

        auto my_policy5 = TestUtils::make_device_policy<class Scan5>(my_policy);
        auto fut3b = dpl::experimental::inclusive_scan_async(my_policy5, dpl::begin(x), dpl::end(x), dpl::begin(y), std::plus<int>(), gamma);
        auto result3b = fut3b.get().get_buffer().get_host_access(sycl::read_only)[n-1];
        EXPECT_TRUE(result3b == expected1, "wrong effect from async scan test (IIIb) with sycl buffer");

        auto my_policy6 = TestUtils::make_device_policy<class Scan6>(my_policy);
        auto fut3c = dpl::experimental::inclusive_scan_async(my_policy6, dpl::begin(x), dpl::end(x), dpl::begin(y), std::plus<int>(), 1, fut3b);
        auto result3c = fut3c.get().get_buffer().get_host_access(sycl::read_only)[n-1];
        EXPECT_TRUE(result3c == (expected1 + 1), "wrong effect from async scan test (IIIc) with sycl buffer");

        // exclusive (2 overloads)
        auto my_policy7 = TestUtils::make_device_policy<class Scan7>(my_policy);
        auto delta = dpl::experimental::exclusive_scan_async(my_policy7, dpl::begin(x), dpl::end(x), dpl::begin(y), 0);
        auto result4 = delta.get().get_buffer().get_host_access(sycl::read_only)[n-1];
        EXPECT_TRUE(result4 == expected2, "wrong effect from async scan test (IV) with sycl buffer");

        auto my_policy8 = TestUtils::make_device_policy<class Scan8>(my_policy);
        auto fut4b = dpl::experimental::exclusive_scan_async(my_policy8, dpl::begin(x), dpl::end(x), dpl::begin(y), 1, std::plus<int>(), delta);
        auto result4b = fut4b.get().get_buffer().get_host_access(sycl::read_only)[n-1];
        EXPECT_TRUE(result4b == (expected2 + 1), "wrong effect from async scan test (IV) with sycl buffer");

        oneapi::dpl::experimental::wait_for_all(alpha,beta,gamma,delta);
    }
}
#endif

int main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    test_with_buffers();
#endif
    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
