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
        auto new_first = std::shift_right(std::ranges::begin(r), std::ranges::end(r), shift);
        return std::borrowed_subrange_t<decltype(r)>{new_first, std::ranges::end(r)};
    };

    const int test_sz = (1<<18) + 739; // 256K+
    test_range_algo<0>{test_sz}(dpl_ranges::shift_right, checker, test_sz/4);
    test_range_algo<1>{test_sz}(dpl_ranges::shift_right, checker, 3 * test_sz/4);
    test_range_algo<2>{small_size}(dpl_ranges::shift_right, checker, small_size + 2);
    test_range_algo<3>{small_size}(dpl_ranges::shift_right, checker, 0);
    test_range_algo<4>{small_size}(dpl_ranges::shift_right, checker, -2);
#endif //_ENABLE_STD_RANGES_TESTING

    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
