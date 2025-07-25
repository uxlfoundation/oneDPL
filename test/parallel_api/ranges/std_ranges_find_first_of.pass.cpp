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

    auto find_first_of_checker = TEST_PREPARE_CALLABLE(std::ranges::find_first_of);
    

    test_range_algo<0, int, data_in_in>{medium_size}(dpl_ranges::find_first_of, find_first_of_checker, binary_pred);
    test_range_algo<1, int, data_in_in>{}(dpl_ranges::find_first_of, find_first_of_checker, binary_pred_const);
    test_range_algo<2, int, data_in_in>{}(dpl_ranges::find_first_of, find_first_of_checker, binary_pred, proj);
    test_range_algo<3, P2, data_in_in>{}(dpl_ranges::find_first_of, find_first_of_checker, binary_pred, &P2::x, &P2::x);
    test_range_algo<4, P2, data_in_in>{}(dpl_ranges::find_first_of, find_first_of_checker, binary_pred, &P2::proj, &P2::proj);

    //false result test case; data generator is a 'gen(i)', so std::identity produces 0, 1, 2, ...
    auto gen_negative = [](auto i) { return -i; };
    test_range_algo<5, int, data_in_in, decltype(gen_negative)>{medium_size}(dpl_ranges::find_first_of, find_first_of_checker, binary_pred);
#endif //_ENABLE_STD_RANGES_TESTING

    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
