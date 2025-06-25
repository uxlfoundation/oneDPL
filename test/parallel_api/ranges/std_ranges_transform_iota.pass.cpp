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
    const char* err_msg = "Wrong effect algo transform with unsized ranges.";

    const int n = medium_size;
    std::ranges::iota_view view1(0, n); //size range
    std::ranges::iota_view view2(0, std::unreachable_sentinel_t{}); //unsized

    std::vector<int> res(n), expected(n);
    std::ranges::transform(view1, view2, expected.begin(), binary_f, proj, proj);

    dpl_ranges::transform(oneapi::dpl::execution::seq, view1, view2, res, binary_f, proj, proj);
    EXPECT_EQ_N(expected.begin(), res.begin(), n, err_msg);

    dpl_ranges::transform(oneapi::dpl::execution::unseq, view1, view2, res, binary_f, proj, proj);
    EXPECT_EQ_N(expected.begin(), res.begin(), n, err_msg);

    dpl_ranges::transform(oneapi::dpl::execution::par, view1, view2, res, binary_f, proj, proj);
    EXPECT_EQ_N(expected.begin(), res.begin(), n, err_msg);

    dpl_ranges::transform(oneapi::dpl::execution::par_unseq, view1, view2, res, binary_f);
    std::ranges::transform(view1, view2, expected.begin(), binary_f);
    EXPECT_EQ_N(expected.begin(), res.begin(), n, err_msg);

    //view1 <-> view2
    dpl_ranges::transform(oneapi::dpl::execution::par_unseq, view2, view1, res, binary_f);
    std::ranges::transform(view2, view1, expected.begin(), binary_f);
    EXPECT_EQ_N(expected.begin(), res.begin(), n, err_msg);

#endif //_ENABLE_STD_RANGES_TESTING

    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
