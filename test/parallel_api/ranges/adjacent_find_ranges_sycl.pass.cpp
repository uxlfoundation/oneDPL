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

#include "support/test_config.h"

#include <oneapi/dpl/execution>

#if _ENABLE_RANGES_TESTING
#include <oneapi/dpl/ranges>
#endif

#include "support/utils.h"
#include "support/utils_invoke.h" // CREATE_NEW_POLICY

#include <iostream>

#if _ENABLE_RANGES_TESTING
template <typename Policy>
void test(Policy&& exec)
{
    constexpr int n = 10;
    int data[n] = {5, 6, 7, 3, 4, 5, 6, 7, 8, 9};

    constexpr int idx = 5;
    data[idx] = -1, data[idx + 1] = -1;

    int res1 = -1, res2 = -1;
    using namespace oneapi::dpl::experimental::ranges;
    {
        sycl::buffer<int> A(data, sycl::range<1>(n));

        res1 = adjacent_find(CREATE_NEW_POLICY(exec, 0), views::all_read(A));
        res2 = adjacent_find(CREATE_NEW_POLICY(exec, 1), A, [](auto a, auto b) {return a == b;});
    }

    //check result
    EXPECT_TRUE(res1 == idx, "wrong effect from 'adjacent_find', sycl ranges");
    EXPECT_TRUE(res2 == idx, "wrong effect from 'adjacent_find' with predicate, sycl ranges");
}
#endif // _ENABLE_RANGES_TESTING

std::int32_t
main()
{
#if _ENABLE_RANGES_TESTING

    auto q = TestUtils::get_test_queue();

    auto policy = TestUtils::make_new_policy<class Kernel1>(q);
    test(policy);

    const auto& policy_ref = policy;
    test(policy_ref);

#endif //_ENABLE_RANGES_TESTING

    return TestUtils::done(_ENABLE_RANGES_TESTING);
}
