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

    auto move_checker = [](std::ranges::random_access_range auto&& r_in,
                           std::ranges::random_access_range auto&& r_out)
    {
        return std::ranges::min(std::forward<decltype(r_in)>(r_in),  std::ranges::begin(r_out));
    };

    test_range_algo<0, int, data_in_out_lim>{big_sz}(dpl_ranges::move,  move_checker);

#endif //_ENABLE_STD_RANGES_TESTING

    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
