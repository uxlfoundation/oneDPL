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

#if _ENABLE_STD_RANGES_TESTING
namespace test_std_ranges
{
template<>
constexpr int calc_res_size<std::remove_cvref_t<decltype(oneapi::dpl::ranges::remove)>>(int n, int res_n)
{ 
    return n - res_n; //res_n - number of elements to be removed
}
} //test_std_ranges
#endif

std::int32_t
main()
{
#if _ENABLE_STD_RANGES_TESTING
    using namespace test_std_ranges;
    namespace dpl_ranges = oneapi::dpl::ranges;

    auto remove_if_checker = TEST_PREPARE_CALLABLE(std::ranges::remove_if);

    test_range_algo<0>{big_sz}(dpl_ranges::remove_if, remove_if_checker, pred);
    test_range_algo<1>{}(dpl_ranges::remove_if, remove_if_checker, pred, proj);
    test_range_algo<2, P2>{}(dpl_ranges::remove_if, remove_if_checker, pred, &P2::x);
    test_range_algo<3, P2>{}(dpl_ranges::remove_if, remove_if_checker, pred, &P2::proj);
#endif //_ENABLE_STD_RANGES_TESTING

    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
