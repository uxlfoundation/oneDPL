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

    auto reverse_copy_checker = [](std::ranges::random_access_range auto&& r_in,
                                   std::ranges::random_access_range auto&& r_out)
    {
        const auto size = std::ranges::min(std::ranges::size(r_in), std::ranges::size(r_out));

        std::ranges::reverse_copy(std::ranges::take_view(r_in, size), std::ranges::take_view(r_out, size).begin());

        using ret_type = std::ranges::in_in_out_result<std::ranges::borrowed_iterator_t<decltype(r_in)>,
                                                       std::ranges::borrowed_iterator_t<decltype(r_in)>,
                                                       std::ranges::borrowed_iterator_t<decltype(r_out)>>;

        auto last_in = std::ranges::begin(r_in) + std::ranges::size(r_in);
        auto stop_in = std::ranges::begin(r_in) + (std::ranges::size(r_in) - size);
        auto stop_out = std::ranges::begin(r_out) + size;
        return ret_type{last_in, stop_in, stop_out};
    };

    test_range_algo<0, int, data_in_out_lim>{big_sz}(dpl_ranges::reverse_copy, reverse_copy_checker);
#endif //_ENABLE_STD_RANGES_TESTING

    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
