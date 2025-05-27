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
#include <oneapi/dpl/algorithm>

#if _ENABLE_RANGES_TESTING
#include <oneapi/dpl/ranges>
#endif

#include "support/utils.h"

#include <iostream>

std::int32_t
main()
{
#if _ENABLE_RANGES_TESTING
    constexpr int max_n = 10;

    using namespace oneapi::dpl::experimental::ranges;

    sycl::buffer<int> A(max_n);

    auto iota = views::iota(0, max_n);
    //the name nano::ranges::copy is not injected into oneapi::dpl::experimental::ranges namespace
    __nanorange::nano::ranges::copy(iota, views::host_all(A).begin());
    reverse(TestUtils::dpcpp_policy(), A);

    for(auto v: views::host_all(A))
        ::std::cout << v << " ";
    ::std::cout << ::std::endl;
    //check result
    EXPECT_EQ_RANGES(iota | views::reverse, views::host_all(A), "wrong effect from reverse");

#endif //_ENABLE_RANGES_TESTING
    return TestUtils::done(_ENABLE_RANGES_TESTING);
}
