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

    auto generate_checker = TEST_PREPARE_CALLABLE(std::ranges::generate);
    auto gen_const = []() { return -3; };
    int seed = 6;
    auto gen = [seed]() mutable { return seed; };

    test_range_algo<0>{big_sz}(dpl_ranges::generate, generate_checker, gen_const);
    test_range_algo<1>{}(dpl_ranges::generate, generate_checker, gen);
    test_range_algo<2>{}(dpl_ranges::generate, generate_checker, gen);
#endif //_ENABLE_STD_RANGES_TESTING

    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
