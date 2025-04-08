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
    const char* err_msg = "Wrong effect algo mismatch with unsized ranges.";

    const int n = medium_size;
    std::ranges::iota_view view1(0, n); //size range
    std::ranges::iota_view view2(0, std::unreachable_sentinel_t{}); //unsized

    auto [ex_res1, ex_res2] = std::ranges::mismatch(view1, view2, binary_pred, proj, proj);

    {
    auto [res1, res2] = dpl_ranges::mismatch(oneapi::dpl::execution::seq, view1, view2, binary_pred, proj, proj);
    EXPECT_TRUE(ex_res1 == res1, err_msg);
    EXPECT_TRUE(ex_res2 == res2, err_msg);
    }

    {
    auto [res1, res2] = dpl_ranges::mismatch(oneapi::dpl::execution::unseq, view1, view2, binary_pred, proj, proj);
    EXPECT_TRUE(ex_res1 == res1, err_msg);
    EXPECT_TRUE(ex_res2 == res2, err_msg);
    }

    {
    auto [res1, res2] = dpl_ranges::mismatch(oneapi::dpl::execution::par, view1, view2, binary_pred, proj, proj);
    EXPECT_TRUE(ex_res1 == res1, err_msg);
    EXPECT_TRUE(ex_res2 == res2, err_msg);
    }

    {
    auto [ex_res1, ex_res2] = std::ranges::mismatch(view2, view1, binary_pred, proj, proj);
    auto [res1, res2] = dpl_ranges::mismatch(oneapi::dpl::execution::par_unseq, view2, view1, binary_pred, proj, proj);
    EXPECT_TRUE(ex_res1 == res1, err_msg);
    EXPECT_TRUE(ex_res2 == res2, err_msg);
    }

#if TEST_DPCPP_BACKEND_PRESENT
    auto exec = dpcpp_policy();
    {
    auto [res1, res2] = dpl_ranges::mismatch(exec, view1, view2, binary_pred, proj, proj);
    EXPECT_TRUE(ex_res1 == res1, err_msg);
    EXPECT_TRUE(ex_res2 == res2, err_msg);
    }
#endif //TEST_DPCPP_BACKEND_PRESENT
#endif //_ENABLE_STD_RANGES_TESTING

    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
