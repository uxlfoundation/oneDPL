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

template<typename>
struct print_type;

std::int32_t
main()
{
#if _ENABLE_STD_RANGES_TESTING

    namespace dpl_ranges = oneapi::dpl::ranges;
    const char* err_msg = "Wrong effect algo transform with zip_view.";

    constexpr int max_n = 10;
    int data[max_n] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

    auto zip_view = my::zip(data, std::views::iota(0, max_n)) | std::views::take(5);
    std::ranges::for_each(zip_view, test_std_ranges::f_mutuable, [](const auto& val) { return std::get<1>(val); });

    test_std_ranges::call_with_host_policies(dpl_ranges::for_each, zip_view, test_std_ranges::f_mutuable, [](const auto& val) { return std::get<1>(val); });
    //EXPECT_EQ_N(expected.begin(), res.begin(), n, err_msg);

    dpl_ranges::for_each(test_std_ranges::dpcpp_policy(), zip_view, test_std_ranges::f_mutuable, [](const auto& val) { return std::get<1>(val); });

    auto zip_view_sort = my::zip(data, data);

    auto it = zip_view_sort.begin();
    std::sort(zip_view_sort.begin(), zip_view_sort.begin() + 5, [](const auto& val1, const auto& val2) { return std::get<0>(val1) < std::get<0>(val2); });
    std::ranges::sort(zip_view_sort, std::greater{}, [](auto&& val) { return std::get<0>(val); });

    static_assert(std::ranges::random_access_range<decltype(zip_view_sort)>);
    static_assert(std::random_access_iterator<decltype(zip_view_sort.begin())>);
    //dpl_ranges::sort(oneapi::dpl::execution::seq, zip_view_sort, std::greater{}, [](auto&& val) { return std::get<0>(val); });

    //test_std_ranges::call_with_host_policies(dpl_ranges::sort, zip_view_sort, test_std_ranges::binary_pred, [](const auto& val) { return std::get<0>(val); });


#endif //_ENABLE_STD_RANGES_TESTING

    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
