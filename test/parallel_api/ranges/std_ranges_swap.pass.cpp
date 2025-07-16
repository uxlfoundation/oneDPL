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

    auto swap_ranges_checker = [](std::ranges::random_access_range auto&& r1,
                           std::ranges::random_access_range auto&& r2)
    {
        const auto size = std::ranges::min(std::ranges::size(r1), std::ranges::size(r2));

        std::ranges::copy(std::ranges::take_view(r1, size), std::ranges::take_view(r2, size).begin());

        using ret_type = std::ranges::swap_ranges_result<std::ranges::borrowed_iterator_t<decltype(r1)>,
            std::ranges::borrowed_iterator_t<decltype(r2)>>;

        return ret_type{std::ranges::begin(r1) + size, std::ranges::begin(r2) + size};
    };

    test_range_algo<0, int, data_in_out_lim>{big_sz}(dpl_ranges::swap_ranges,  swap_ranges_checker);
#endif //_ENABLE_STD_RANGES_TESTING

    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
