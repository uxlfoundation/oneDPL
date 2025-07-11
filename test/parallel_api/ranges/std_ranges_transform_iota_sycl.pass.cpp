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

#include "support/utils_invoke.h" // for CLONE_TEST_POLICY macro

#if _ENABLE_STD_RANGES_TESTING && TEST_DPCPP_BACKEND_PRESENT
template <typename Policy>
void
test_impl(Policy&& exec)
{
    using namespace test_std_ranges;
    namespace dpl_ranges = oneapi::dpl::ranges;
    const char* err_msg = "Wrong effect algo transform with unsized ranges.";

    const int n = big_size;
    std::ranges::iota_view view1(0, n); //size range
    std::ranges::iota_view view2(0, std::unreachable_sentinel_t{}); //unsized

    std::vector<int> src(n), expected(n);
    std::ranges::transform(view1, view2, expected.begin(), binary_f, proj, proj);

    usm_subrange<int> cont_out(CLONE_TEST_POLICY_IDX(exec, 0), src.data(), n);
    auto res = cont_out();

    dpl_ranges::transform(CLONE_TEST_POLICY_IDX(exec, 1), view1, view2, res, binary_f, proj, proj);
    EXPECT_EQ_N(expected.begin(), res.begin(), n, err_msg);

    //view1 <-> view2
    std::ranges::transform(view2, view1, expected.begin(), binary_f, proj, proj);
    dpl_ranges::transform(CLONE_TEST_POLICY_IDX(exec, 2), view2, view1, res, binary_f, proj, proj);
    EXPECT_EQ_N(expected.begin(), res.begin(), n, err_msg);
}
#endif // _ENABLE_STD_RANGES_TESTING && TEST_DPCPP_BACKEND_PRESENT

std::int32_t
main()
{
    bool bProcessed = false;

#if _ENABLE_STD_RANGES_TESTING && TEST_DPCPP_BACKEND_PRESENT

    auto policy = TestUtils::get_dpcpp_test_policy();
    test_impl(policy);

#if TEST_CHECK_COMPILATION_WITH_DIFF_POLICY_VAL_CATEGORY
    TestUtils::check_compilation(policy, [](auto&& policy) { test_impl(std::forward<decltype(policy)>(policy)); });
#endif

    bProcessed = true;

#endif // _ENABLE_STD_RANGES_TESTING && TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(bProcessed);
}
