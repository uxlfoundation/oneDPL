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

std::int32_t
main()
{
#if _ENABLE_STD_RANGES_TESTING
    using namespace test_std_ranges;
    namespace dpl_ranges = oneapi::dpl::ranges;

    const int n = medium_size;

    //transform view
    test_range_algo<0>{n}.test_view(std::views::transform([](const auto a) { return a*2; }),
        dpl_ranges::find_if, std::ranges::find_if, pred, proj);

    //reverse view
    test_range_algo<1>{n}.test_view(std::views::reverse, dpl_ranges::sort, std::ranges::sort, std::less{});

    //take view
    test_range_algo<2>{n}.test_view(std::views::take(n/2), dpl_ranges::count_if, std::ranges::count_if, pred, proj);

    //drop view
    test_range_algo<3>{n}.test_view(std::views::drop(n/2), dpl_ranges::count_if, std::ranges::count_if, pred, proj);

    //NOTICE: std::ranges::views::all, std::ranges::subrange, std::span are tested implicitly within the 'test_range_algo' test engine.
#endif //_ENABLE_STD_RANGES_TESTING

    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}

