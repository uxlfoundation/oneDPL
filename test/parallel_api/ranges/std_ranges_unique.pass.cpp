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
namespace dpl_ranges = oneapi::dpl::ranges;

template<>
constexpr std::pair<int, int>
test_std_ranges::range_to_verify<std::remove_cvref_t<decltype(dpl_ranges::unique)>>(int total_size, int result_size)
{ 
    return {0, total_size - result_size}; // in the result are the elements to remove
}
#endif

int
main()
{
#if _ENABLE_STD_RANGES_TESTING
    using namespace test_std_ranges;

    auto unique_checker = TEST_PREPARE_CALLABLE(std::ranges::unique);

    // input generator with a fair chance of repeating the previous value
    auto repeat_sometimes = [](auto i) {
        static decltype(i) last = 0;
        if (i == 0)
            last = 0; // reset
        else if (i%7 > 0 && (last + i - 1)%3 == 0)
            last = i;
        return last;
    };
    using repeating_gen = decltype(repeat_sometimes);

    auto equal_tens = [](auto i, auto j) { return i/10 == j/10; };

    test_range_algo<0>{get_scan_big_sz()}(dpl_ranges::unique, unique_checker);
    test_range_algo<1, int, data_in, repeating_gen>{}(dpl_ranges::unique, unique_checker, std::ranges::equal_to{});
    test_range_algo<2>{small_size}(dpl_ranges::unique, unique_checker, std::not_equal_to{});
    test_range_algo<3>{}(dpl_ranges::unique, unique_checker, equal_tens, proj);
    test_range_algo<4, P2, data_in, repeating_gen>{}(dpl_ranges::unique, unique_checker, std::ranges::equal_to{}, &P2::x);
    test_range_algo<5, P2>{}(dpl_ranges::unique, unique_checker, std::ranges::equal_to{}, &P2::proj);
#endif //_ENABLE_STD_RANGES_TESTING

    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
