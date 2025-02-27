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
#if _ENABLE_STD_RANGES_TESTING && TEST_DPCPP_BACKEND_PRESENT
    using namespace test_std_ranges;
    namespace dpl_ranges = oneapi::dpl::ranges;
    const char* err_msg = "Wrong effect algo transform with unsized ranges.";

    const int n = big_size;
    std::ranges::iota_view view1(0, n); //size range
    std::ranges::iota_view view2(0, std::unreachable_sentinel_t{}); //unsized

    std::vector<int> src(n), expected(n);
    std::ranges::transform(view1, view2, expected.begin(), binary_f, proj, proj);

    auto exec = dpcpp_policy();
    using Policy = decltype(exec);
    auto exec1 = TestUtils::make_new_policy<TestUtils::new_kernel_name<Policy, 0>>(exec);
    auto exec2 = TestUtils::make_new_policy<TestUtils::new_kernel_name<Policy, 1>>(exec);

    usm_subrange<int> cont_out(exec, src.data(), n);
    auto res = cont_out();

    dpl_ranges::transform(exec1, view1, view2, res, binary_f, proj, proj);
    EXPECT_EQ_N(expected.begin(), res.begin(), n, err_msg);

    //view1 <-> view2
    std::ranges::transform(view2, view1, expected.begin(), binary_f, proj, proj);
    dpl_ranges::transform(exec2, view2, view1, res, binary_f, proj, proj);
    EXPECT_EQ_N(expected.begin(), res.begin(), n, err_msg);

#endif //_ENABLE_STD_RANGES_TESTING

    return TestUtils::done(_ENABLE_STD_RANGES_TESTING && TEST_DPCPP_BACKEND_PRESENT);
}
