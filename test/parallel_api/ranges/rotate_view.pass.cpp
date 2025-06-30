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

#include <iostream>

std::int32_t
main()
{
#if _ENABLE_RANGES_TESTING
    constexpr int max_n = 10;
    int data[max_n]     = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    int expected[max_n] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    int rotate_val = 6;

    using namespace oneapi::dpl::experimental::ranges;

    //the name nano::ranges::views::all is not injected into oneapi::dpl::experimental::ranges namespace
    auto view1 = __nanorange::nano::ranges::views::all(data) | views::rotate(rotate_val);
    auto view2 = views::rotate(__nanorange::nano::ranges::views::all(data), rotate_val);

    //check result
    ::std::rotate_copy(data, data + rotate_val, data + max_n, expected);

    //check aasigment
    view1[0] = -1;
    expected[0] = -1;

    EXPECT_EQ_N(expected, view1.begin(), max_n, "wrong result from rotate view, a pipe call");
    EXPECT_EQ_N(expected, view2.begin(), max_n, "wrong result from rotate view, a single CPO call");

#endif //_ENABLE_RANGES_TESTING
    return TestUtils::done(_ENABLE_RANGES_TESTING);
}
