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

struct MultiplyByTwo
{
    template <typename T>
    auto
    operator()(T x) const
    {
        return x * 2;
    }
};

void
test_impl_host()
{
    using namespace test_std_ranges;
    namespace dpl_ranges = oneapi::dpl::ranges;

    const int n = medium_size;

    //transform view
    test_range_algo<0>{n}.test_view_host(std::views::transform(MultiplyByTwo{}),
        dpl_ranges::find_if, std::ranges::find_if, pred, proj);

    //reverse view
    test_range_algo<1>{n}.test_view_host(std::views::reverse, dpl_ranges::sort, std::ranges::sort, std::less{});

    //take view
    test_range_algo<2>{n}.test_view_host(std::views::take(n/2), dpl_ranges::count_if, std::ranges::count_if, pred, proj);

    //drop view
    test_range_algo<3>{n}.test_view_host(std::views::drop(n/2), dpl_ranges::count_if, std::ranges::count_if, pred, proj);

    //NOTICE: std::ranges::views::all, std::ranges::subrange, std::span are tested implicitly within the 'test_range_algo' test engine.
}

#if TEST_DPCPP_BACKEND_PRESENT
template <typename Policy>
void
test_impl_hetero(Policy&& exec)
{
    using namespace test_std_ranges;
    namespace dpl_ranges = oneapi::dpl::ranges;

    const int n = medium_size;

    //transform view
    test_range_algo<0>{n}.test_view_hetero(CLONE_TEST_POLICY(exec), std::views::transform(MultiplyByTwo{}),
        dpl_ranges::find_if, std::ranges::find_if, pred, proj);

    //reverse view
    test_range_algo<1>{n}.test_view_hetero(CLONE_TEST_POLICY(exec), std::views::reverse, dpl_ranges::sort, std::ranges::sort, std::less{});

    //take view
    test_range_algo<2>{n}.test_view_hetero(CLONE_TEST_POLICY(exec), std::views::take(n/2), dpl_ranges::count_if, std::ranges::count_if, pred, proj);

#if !_PSTL_LIBSTDCXX_XPU_DROP_VIEW_BROKEN
    //drop view
    test_range_algo<3>{n}.test_view_hetero(CLONE_TEST_POLICY(exec), std::views::drop(n/2), dpl_ranges::count_if, std::ranges::count_if, pred, proj);
#endif

    //NOTICE: std::ranges::views::all, std::ranges::subrange, std::span are tested implicitly within the 'test_range_algo' test engine.
}
#endif // TEST_DPCPP_BACKEND_PRESENT
#endif // _ENABLE_STD_RANGES_TESTING

std::int32_t
main()
{
#if _ENABLE_STD_RANGES_TESTING

    test_impl_host();

#if TEST_DPCPP_BACKEND_PRESENT
    auto policy = TestUtils::get_dpcpp_test_policy();
    test_impl_hetero(policy);

#if TEST_CHECK_COMPILATION_WITH_DIFF_POLICY_VAL_CATEGORY
    TestUtils::check_compilation(policy, [](auto&& policy) { test_impl_hetero(std::forward<decltype(policy)>(policy)); });
#endif
#endif // TEST_DPCPP_BACKEND_PRESENT
#endif //_ENABLE_STD_RANGES_TESTING

    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}

