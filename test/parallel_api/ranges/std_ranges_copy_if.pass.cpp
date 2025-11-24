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

    auto copy_if_checker = []<typename Proj = std::identity>(std::ranges::random_access_range auto&& r_in,
                           std::ranges::random_access_range auto&& r_out, auto pred, Proj proj = {})
    {
        using ret_type = std::ranges::copy_if_result<std::ranges::borrowed_iterator_t<decltype(r_in)>,
            std::ranges::borrowed_iterator_t<decltype(r_out)>>;

        auto in = std::ranges::begin(r_in);
        auto out = std::ranges::begin(r_out);
        std::size_t i = 0, j = 0;
        for(; i < std::ranges::size(r_in); ++i)
        {
             if (std::invoke(pred, std::invoke(proj, in[i])))
             {
                 if (j < std::ranges::size(r_out))
                     out[j++] = in[i];
                 else
                     break;
             }
        }
        return ret_type{in + i, out + j};
    };

#if TEST_DPCPP_BACKEND_PRESENT
    constexpr TestDataMode test_mode = TestDataMode::data_in_out;
#else
    constexpr TestDataMode test_mode = TestDataMode::data_in_out_lim;
#endif

    test_range_algo<0, int, test_mode>{217}(dpl_ranges::copy_if, copy_if_checker, pred);
    test_range_algo<1, int, test_mode>{1234}(dpl_ranges::copy_if, copy_if_checker, select_many);
    test_range_algo<2, int, test_mode>{}(dpl_ranges::copy_if, copy_if_checker, select_many, proj);
    test_range_algo<3, P2, test_mode>{}(dpl_ranges::copy_if, copy_if_checker, pred, &P2::x);
    test_range_algo<4, P2, test_mode>{}(dpl_ranges::copy_if, copy_if_checker, pred, &P2::proj);
    test_range_algo<5, int, test_mode>{big_sz}(dpl_ranges::copy_if, copy_if_checker, pred);
#endif //_ENABLE_STD_RANGES_TESTING

    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
