// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "std_ranges_test.h"

int
main()
{
#if _ENABLE_STD_RANGES_TESTING
    using namespace test_std_ranges;
    namespace dpl_ranges = oneapi::dpl::ranges;

    auto checker = [](std::ranges::random_access_range auto&& r, int shift)
    {
        auto new_last = std::shift_left(std::ranges::begin(r), std::ranges::end(r), shift);
        return std::ranges::borrowed_subrange_t<decltype(r)>{std::ranges::begin(r), new_last};
    };

    const int test_sz = (1<<18) + 953; // 256K+
    test_range_algo<0>{test_sz}(dpl_ranges::shift_left, checker, test_sz/4);
    test_range_algo<1>{test_sz}(dpl_ranges::shift_left, checker, 3 * test_sz/4);
    test_range_algo<2>{small_size}(dpl_ranges::shift_left, checker, small_size + 2);
    test_range_algo<3>{small_size}(dpl_ranges::shift_left, checker, 0);
#endif //_ENABLE_STD_RANGES_TESTING

    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
