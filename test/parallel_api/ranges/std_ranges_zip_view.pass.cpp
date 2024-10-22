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
#include <oneapi/dpl/ranges>

#if _ENABLE_STD_RANGES_TESTING
#include <vector>

void test_zip_view_base_op()
{
    namespace dpl_ranges = oneapi::dpl::ranges;

    constexpr int max_n = 100;
    std::vector<int> vec1(max_n);
    std::vector<int> vec2(max_n/2);

    auto zip_view = dpl_ranges::zip(vec1, vec2);

    static_assert(std::random_access_iterator<decltype(zip_view.begin())>);
    static_assert(std::sentinel_for<decltype(zip_view.end()), decltype(zip_view.begin())>);

    static_assert(std::random_access_iterator<decltype(zip_view.cbegin())>);
    static_assert(std::sentinel_for<decltype(zip_view.cend()), decltype(zip_view.cbegin())>);

    EXPECT_TRUE(zip_view.end() - zip_view.begin() == max_n/2,
        "Difference operation between an iterator and a sentinel (zip_view) returns a wrong result.");

    EXPECT_TRUE(zip_view[2] == *(zip_view.begin() + 2), 
        "Subscription or dereferencing operation for zip_view returns a wrong result.");

    EXPECT_TRUE(std::ranges::size(zip_view) == max_n/2, "zip_view::size method returns a wrong result.");
    EXPECT_TRUE((bool)zip_view, "zip_view::operator bool() method returns a wrong result.");
}
#endif //_ENABLE_STD_RANGES_TESTING

std::int32_t
main()
{
#if _ENABLE_STD_RANGES_TESTING

    test_zip_view_base_op();

    namespace dpl_ranges = oneapi::dpl::ranges;

    constexpr int max_n = 10;
    int data[max_n] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

    auto zip_view = dpl_ranges::zip(data, std::views::iota(0, max_n)) | std::views::take(5);
    std::ranges::for_each(zip_view, test_std_ranges::f_mutuable, [](const auto& val) { return std::get<1>(val); });

    test_std_ranges::call_with_host_policies(dpl_ranges::for_each, zip_view, test_std_ranges::f_mutuable, [](const auto& val) { return std::get<1>(val); });

#if TEST_DPCPP_BACKEND_PRESENT
    dpl_ranges::for_each(test_std_ranges::dpcpp_policy(), zip_view, test_std_ranges::f_mutuable, [](const auto& val) { return std::get<1>(val); });
#endif

    auto zip_view_sort = dpl_ranges::zip(data, data);

    std::sort(zip_view_sort.begin(), zip_view_sort.begin() + max_n, [](const auto& val1, const auto& val2) { return std::get<0>(val1) > std::get<0>(val2); });
    for(int i = 0; i < max_n; ++i)
        assert(std::get<0>(zip_view_sort[i]) == max_n - 1 - i);

    std::ranges::sort(zip_view_sort, std::less{}, [](auto&& val) { return std::get<0>(val); });
    for(int i = 0; i < max_n; ++i)
        assert(std::get<0>(zip_view_sort[i]) == i);

    static_assert(std::ranges::random_access_range<decltype(zip_view_sort)>);
    static_assert(std::random_access_iterator<decltype(zip_view_sort.begin())>);

    test_std_ranges::call_with_host_policies(dpl_ranges::sort, zip_view_sort, std::greater{}, [](const auto& val) { return std::get<0>(val); });
    for(int i = 0; i < max_n; ++i)
        assert(std::get<0>(zip_view_sort[i]) == max_n - 1 - i);

#endif //_ENABLE_STD_RANGES_TESTING

    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
