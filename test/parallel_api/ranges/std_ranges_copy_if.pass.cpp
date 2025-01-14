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

    auto copy_if_checker = [](std::ranges::random_access_range auto&& r_in,
                           std::ranges::random_access_range auto&& r_out, auto pred, auto proj)
    {
        using ret_type = std::ranges::copy_if_result<std::ranges::borrowed_iterator_t<decltype(r_in)>,
            std::ranges::borrowed_iterator_t<decltype(r_out)>>;

        auto it_in = std::ranges::begin(r_in);
        auto it_out = std::ranges::begin(r_out);
        for(; it_in != std::ranges::end(r_in) && it_out != std::ranges::end(r_out); ++it_in)
        {
             if (std::invoke(pred, std::invoke(proj, *it_in)))
             {
                 *it_out = *it_in;
                 ++it_out;
             }
        }
        return ret_type{it_in, it_out};
    };

    test_range_algo<0, int, data_in_out>{big_sz}(dpl_ranges::copy_if,  copy_if_checker, pred, std::identity{});
    test_range_algo<1, int, data_in_out>{}(dpl_ranges::copy_if,  copy_if_checker, pred, proj);
    test_range_algo<2, P2, data_in_out>{}(dpl_ranges::copy_if,  copy_if_checker, pred, &P2::x);
    test_range_algo<3, P2, data_in_out>{}(dpl_ranges::copy_if,  copy_if_checker, pred, &P2::proj);
#endif //_ENABLE_STD_RANGES_TESTING

    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
