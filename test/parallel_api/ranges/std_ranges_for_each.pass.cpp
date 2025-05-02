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

    //A checker below modifies a return type; a range based version with policy has another return type.
    auto for_each_checker = [](auto&&... args) {
        auto res = std::ranges::for_each(std::forward<decltype(args)>(args)...);
        return res.in;
    };

    test_range_algo<0>{big_sz}(dpl_ranges::for_each, for_each_checker, f_mutuable);
    test_range_algo<1>{}(dpl_ranges::for_each, for_each_checker, f_mutuable, proj_mutuable);
    test_range_algo<2, P2>{}(dpl_ranges::for_each, for_each_checker, f_mutuable, &P2::x);
    test_range_algo<3, P2>{}(dpl_ranges::for_each, for_each_checker, f_mutuable, &P2::proj);
#endif //_ENABLE_STD_RANGES_TESTING

    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
