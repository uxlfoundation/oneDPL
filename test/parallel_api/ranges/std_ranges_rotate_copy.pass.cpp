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

#if _ENABLE_STD_RANGES_TESTING
// Wrapper to adjust rotate_copy to the format expected by the test harness
struct
{
    template <typename Policy, std::ranges::random_access_range InRange, std::ranges::random_access_range OutRange>
    auto operator()(Policy&& exec, InRange&& r_in, OutRange&& r_out, int pivot_pos = -1)
    {
        const int in_size = std::ranges::size(r_in);
        auto middle = std::ranges::begin(r_in) + ((pivot_pos < 0)? in_size/3 : std::min<int>(pivot_pos, in_size));

        return oneapi::dpl::ranges::rotate_copy(std::forward<Policy>(exec), std::forward<InRange>(r_in), middle,
                                                std::forward<OutRange>(r_out));
    }
} rotate_copy_tester;
#endif // _ENABLE_STD_RANGES_TESTING

int
main()
{
#if _ENABLE_STD_RANGES_TESTING
    using namespace test_std_ranges;

    auto rotate_copy_checker = [](std::ranges::random_access_range auto&& r_in,
                                  std::ranges::random_access_range auto&& r_out, int pivot_pos = -1)
    {
        auto first_in = std::ranges::begin(r_in);
        auto last_in = std::ranges::end(r_in);
        auto last_out = std::ranges::end(r_out);

        // calculate the pivot point exactly like in the tester above
        const int in_size = std::ranges::size(r_in);
        auto middle = first_in + ((pivot_pos < 0)? in_size/3 : std::min<int>(pivot_pos, in_size));

        auto it1 = middle, it2 = first_in;
        auto it_out = std::ranges::begin(r_out);
        while (it_out != last_out && it1 != last_in)
            *it_out++ = *it1++;
        while (it_out != last_out && it2 != middle)
            *it_out++ = *it2++;

        using ret_type = std::ranges::in_in_out_result<std::ranges::borrowed_iterator_t<decltype(r_in)>,
                                                       std::ranges::borrowed_iterator_t<decltype(r_in)>,
                                                       std::ranges::borrowed_iterator_t<decltype(r_out)>>;
        return ret_type{it1, it2, it_out};
    };

    const int test_sz = 19787;
    test_range_algo<0, int, data_in_out_lim>{big_sz}(rotate_copy_tester, rotate_copy_checker);
    test_range_algo<1, float, data_in_out_lim>{test_sz}(rotate_copy_tester, rotate_copy_checker, test_sz - 1);
    test_range_algo<2, P2, data_in_out_lim>{test_sz/11}(rotate_copy_tester, rotate_copy_checker, 0);
    test_range_algo<3, int, data_in_out_lim>{test_sz/17}(rotate_copy_tester, rotate_copy_checker, 1);
#endif //_ENABLE_STD_RANGES_TESTING

    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
