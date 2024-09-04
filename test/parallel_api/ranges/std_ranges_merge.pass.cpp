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

    //A checker below modifies a return type; a range based version with policy has another return type.
    auto merge_checker = [](std::ranges::random_access_range auto&& r_1,
                                       std::ranges::random_access_range auto&& r_2,
                                       std::ranges::random_access_range auto&& r_out, auto&&... args)
    {
        auto res = std::ranges::merge(std::forward<decltype(r_1)>(r_1), std::forward<decltype(r_2)>(r_2),
            std::ranges::begin(r_out), std::forward<decltype(args)>(args)...);

        using ret_type = std::ranges::merge_result<std::ranges::borrowed_iterator_t<decltype(r_1)>,
            std::ranges::borrowed_iterator_t<decltype(r_2)>, std::ranges::borrowed_iterator_t<decltype(r_out)>>;
        return ret_type{res.in1, res.in2, res.out};
    };

    test_range_algo<0, int, data_in_in_out>{}(dpl_ranges::merge, merge_checker, std::ranges::less{});

    test_range_algo<1, int, data_in_in_out>{}(dpl_ranges::merge, merge_checker, std::ranges::less{}, proj, proj);
    test_range_algo<2, P2, data_in_in_out>{}(dpl_ranges::merge, merge_checker, std::ranges::less{}, &P2::x, &P2::x);
    test_range_algo<3, P2, data_in_in_out>{}(dpl_ranges::merge, merge_checker, std::ranges::less{}, &P2::proj, &P2::proj);

    test_range_algo<4, int, data_in_in_out>{}(dpl_ranges::merge, merge_checker, std::ranges::greater{}, proj, proj);
    test_range_algo<5, P2, data_in_in_out>{}(dpl_ranges::merge, merge_checker, std::ranges::greater{}, &P2::x, &P2::x);
    test_range_algo<6, P2, data_in_in_out>{}(dpl_ranges::merge, merge_checker, std::ranges::greater{}, &P2::proj, &P2::proj);
#endif //_ENABLE_STD_RANGES_TESTING

    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
