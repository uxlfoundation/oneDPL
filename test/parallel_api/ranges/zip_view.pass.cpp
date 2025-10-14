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

#include "support/test_config.h"

#include <oneapi/dpl/execution>

#if _ENABLE_RANGES_TESTING
#    include <oneapi/dpl/ranges>
#endif

#include "support/utils.h"

#include <iostream>

std::int32_t
main()
{
#if _ENABLE_RANGES_TESTING
    constexpr int max_n = 10;
    char data[max_n] = {'b', 'e', 'g', 'f', 'c', 'd', 'a', 'j', 'i', 'h'};
    int key[max_n] = {1, 4, 6, 5, 2, 3, 0, 9, 8, 7};

    using namespace oneapi::dpl::experimental::ranges;

    //the name nano::ranges::views::all is not injected into oneapi::dpl::experimental::ranges namespace
    auto view = __nanorange::nano::views::all(data);
    auto z = zip_view(view, __nanorange::nano::views::all(key));

    //check access
    EXPECT_TRUE(::std::get<0>(z[2]) == 'g', "wrong effect with zip_view");

    int64_t max_int32p2 = (size_t)::std::numeric_limits<int32_t>::max() + 2L;

    auto base_view = views::iota(::std::int64_t(0), max_int32p2);

    //avoiding allocating large amounts of memory, just reusing small data container
    auto transform_data_idx = [&max_n, &data](auto idx) { return data[idx % max_n]; };
    auto data_large_view = views::transform(base_view, transform_data_idx);

    //avoiding allocating large amounts of memory, just reusing small data container
    auto transform_key_idx = [&max_n, &key](auto idx) { return key[idx % max_n]; };
    auto key_large_view = views::transform(base_view, transform_key_idx);

    auto large_z = zip_view(data_large_view, key_large_view);

    //check that zip_view ranges can be larger than a signed 32 bit integer
    size_t i = large_z.size() - 1;

    auto expected_key = key[i % max_n];
    auto actual_key = ::std::get<1>(large_z[i]);
    EXPECT_EQ(expected_key, actual_key, "wrong effect with zip_view bracket operator");

    char expected_data = data[i % max_n];
    char actual_data = ::std::get<0>(large_z[i]);
    EXPECT_EQ(expected_data, actual_data, "wrong effect with zip_view bracket operator");

#endif //_ENABLE_RANGES_TESTING

    return TestUtils::done(_ENABLE_RANGES_TESTING);
}
