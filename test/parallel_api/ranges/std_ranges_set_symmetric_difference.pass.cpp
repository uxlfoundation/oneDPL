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

    // TODO: use data_in_in_out_lim when set_symmetric_difference supports
    // output range not-sufficiently large to hold all the proccessed elements
    // this will also require adding a custom serial implementation of the algorithm into the checker

    auto checker = [](std::ranges::random_access_range auto&& r1,
                      std::ranges::random_access_range auto&& r2,
                      std::ranges::random_access_range auto&& r_out, auto&&... args)
    {
        auto res = std::ranges::set_symmetric_difference(std::forward<decltype(r1)>(r1), std::forward<decltype(r2)>(r2),
                                                         std::ranges::begin(r_out), std::forward<decltype(args)>(args)...);

        using ret_type = std::ranges::set_symmetric_difference_result<std::ranges::borrowed_iterator_t<decltype(r1)>,
                                                                      std::ranges::borrowed_iterator_t<decltype(r2)>,
                                                                      std::ranges::borrowed_iterator_t<decltype(r_out)>>;
        return ret_type{res.in1, res.in2, res.out};
    };

    test_range_algo<0, int, data_in_in_out>{big_sz}(dpl_ranges::set_symmetric_difference, checker);
    test_range_algo<1, int, data_in_in_out>{big_sz}(dpl_ranges::set_symmetric_difference, checker,std::ranges::less{}, proj);

    test_range_algo<2, int, data_in_in_out>{}(dpl_ranges::set_symmetric_difference, checker, std::ranges::less{}, proj, proj);
    //test_range_algo<3,  P2, data_in_in_out>{}(dpl_ranges::set_symmetric_difference, checker, std::ranges::less{}, &P2::x, &P2::x);
    //test_range_algo<4,  P2, data_in_in_out>{}(dpl_ranges::set_symmetric_difference, checker, std::ranges::less{}, &P2::proj, &P2::proj);
#endif //_ENABLE_STD_RANGES_TESTING

    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
