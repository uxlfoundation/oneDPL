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

// MSVC error: SYCL kernel cannot call an undefined function without SYCL_EXTERNAL attribute
// A reason is the vectorised implementation of __std_min_8u(_First, _Last), called from std::sort
// As workaround we suppress the vectorised implementation.
#define _USE_STD_VECTOR_ALGORITHMS 0

#include "std_ranges_test.h"

std::int32_t
main()
{
#if _ENABLE_STD_RANGES_TESTING
    using namespace test_std_ranges;
    namespace dpl_ranges = oneapi::dpl::ranges;

    const int n = medium_size;

    auto zip_proj = [](auto&& val) ->decltype(auto) { return std::get<0>(val); };

    //find_if with zip_view
    test_range_algo<0>{n}.test_view_host(dpl_ranges::views::zip, dpl_ranges::find_if, std::ranges::find_if, pred, zip_proj);

    //sort with zip_view
    test_range_algo<1>{n}.test_view_host(dpl_ranges::views::zip, dpl_ranges::sort, std::ranges::sort, std::less{}, zip_proj);

    //count_if with zip_view
    test_range_algo<2>{n}.test_view_host(dpl_ranges::views::zip, dpl_ranges::count_if, std::ranges::count_if, pred, zip_proj);

#if TEST_DPCPP_BACKEND_PRESENT
    auto exec = TestUtils::get_dpcpp_test_policy();

    //find_if with zip_view
    test_range_algo<0>{n}.test_view_hetero(CLONE_TEST_POLICY(exec), dpl_ranges::views::zip, dpl_ranges::find_if, std::ranges::find_if, pred, zip_proj);

    //sort with zip_view
    test_range_algo<1>{n}.test_view_hetero(CLONE_TEST_POLICY(exec), dpl_ranges::views::zip, dpl_ranges::sort, std::ranges::sort, std::less{}, zip_proj);

    //count_if with zip_view
    test_range_algo<2>{n}.test_view_hetero(CLONE_TEST_POLICY(exec), dpl_ranges::views::zip, dpl_ranges::count_if, std::ranges::count_if, pred, zip_proj);
#endif //TEST_DPCPP_BACKEND_PRESENT


#endif //_ENABLE_STD_RANGES_TESTING

    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}

