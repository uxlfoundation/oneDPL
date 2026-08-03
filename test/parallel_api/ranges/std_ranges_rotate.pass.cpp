// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "std_ranges_test.h"

#if _ENABLE_STD_RANGES_TESTING
// Wrapper to adjust rotate to the format expected by the test harness
struct
{
    template <typename Policy, std::ranges::random_access_range Range>
    auto operator()(Policy&& exec, Range&& r, int pivot_pos = -1)
    {
        const int in_size = std::ranges::size(r);
        auto middle = std::ranges::begin(r) + ((pivot_pos < 0)? in_size/3 : std::min<int>(pivot_pos, in_size));

        return oneapi::dpl::ranges::rotate(std::forward<Policy>(exec), std::forward<Range>(r), middle);
    }
} rotate_tester;
#endif // _ENABLE_STD_RANGES_TESTING

int
main()
{
#if _ENABLE_STD_RANGES_TESTING
    using namespace test_std_ranges;

    auto rotate_checker = [](std::ranges::random_access_range auto&& r, int pivot_pos = -1)
    {
        // calculate the pivot point exactly like in the tester above
        const int in_size = std::ranges::size(r);
        auto middle = std::ranges::begin(r) + ((pivot_pos < 0)? in_size/3 : std::min<int>(pivot_pos, in_size));
        return std::ranges::rotate(std::forward<decltype(r)>(r), middle);
    };

    const int test_sz = 13192;
    test_range_algo<0>{big_sz}(rotate_tester, rotate_checker);
    test_range_algo<1, P2>{test_sz}(rotate_tester, rotate_checker, 0);
    test_range_algo<2>(rotate_tester, rotate_checker, 1);
    test_range_algo<3, float>{test_sz}(rotate_tester, rotate_checker, test_sz - 1);
#endif //_ENABLE_STD_RANGES_TESTING

    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
