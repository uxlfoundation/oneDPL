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
#if TEST_DPCPP_BACKEND_PRESENT
template <typename Policy, typename TView1, typename TView2, typename TRes1, typename TRes2>
void
test_impl(Policy&& exec, TView1&& view1, TView2&& view2, TRes1 ex_res1, TRes2 ex_res2)
{
    using namespace test_std_ranges;
    namespace dpl_ranges = oneapi::dpl::ranges;
    const char* err_msg = "Wrong effect algo mismatch with unsized ranges.";

    auto [res1, res2] = dpl_ranges::mismatch(std::forward<Policy>(exec), std::forward<TView1>(view1), std::forward<TView2>(view2), binary_pred, proj, proj);
    EXPECT_TRUE(ex_res1 == res1, err_msg);
    EXPECT_TRUE(ex_res2 == res2, err_msg);
}
#endif // TEST_DPCPP_BACKEND_PRESENT
#endif // _ENABLE_STD_RANGES_TESTING

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

    auto policy = TestUtils::get_dpcpp_test_policy();
    test_impl(policy, view1, view2, ex_res1, ex_res2);

#if TEST_CHECK_COMPILATION_WITH_DIFF_POLICY_VAL_CATEGORY
    TestUtils::check_compilation(policy, [&](auto&& policy) { test_impl(std::forward<decltype(policy)>(policy), view1, view2, ex_res1, ex_res2); });
#endif
#endif //TEST_DPCPP_BACKEND_PRESENT
#endif //_ENABLE_STD_RANGES_TESTING

    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
