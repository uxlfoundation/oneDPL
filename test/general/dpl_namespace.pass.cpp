// -*- C++ -*-
//===-- dpl_namespace.pass.cpp --------------------------------------------===//
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
#include <oneapi/dpl/iterator>
#include <oneapi/dpl/functional>

#include "support/utils.h"

#include <iostream>
#include <tuple>

#if TEST_DPCPP_BACKEND_PRESENT

class ForEach;
class Transform;
class Scan;

template <typename T>
struct ForEachOp
{
    const int n;

    void operator()(std::tuple<T, T> x) const
    {
        std::get<1>(x) = (2 * std::get<0>(x)) / n;
    }
};

template <typename Policy>
void
test_impl(Policy&& exec)
{
    const int n = 1000;
    const int k = 1000;
    using T = std::uint64_t;

    sycl::buffer<T> key_buf{sycl::range<1>(n)};
    sycl::buffer<T> val_buf{sycl::range<1>(n)};
    sycl::buffer<T> res_buf{sycl::range<1>(k)};

    auto key_first = dpl::begin(key_buf);
    auto val_first = dpl::begin(val_buf);
    auto res_first = dpl::begin(res_buf);
    auto counting_first = dpl::counting_iterator<T>(0);
    auto zip_first = dpl::make_zip_iterator(counting_first, key_first);

    // key_buf = {0,0,...0,1,1,...,1}
    std::for_each(
        CLONE_TEST_POLICY_NAME(exec, ForEach),
        zip_first, zip_first + n,
        ForEachOp<T>{n});

    // val_buf = {0,1,2,...,n-1}
    std::transform(
        CLONE_TEST_POLICY_NAME(exec, Transform),
        counting_first, counting_first + n, val_first, dpl::identity());

    auto result = dpl::inclusive_scan_by_segment(
        CLONE_TEST_POLICY_NAME(exec, Scan),
        key_first, key_first + n, val_first, res_first);

    EXPECT_EQ(k, result - res_first, "size of keys output is not valid");
}

#endif

int main()
{
#if TEST_DPCPP_BACKEND_PRESENT

    auto policy = TestUtils::get_dpcpp_test_policy();
    test_impl(policy);

    TestUtils::check_compilation(policy, [](auto&& policy) { test_impl(std::forward<decltype(policy)>(policy)); });

#endif

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
