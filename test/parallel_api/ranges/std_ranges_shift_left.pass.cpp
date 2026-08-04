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

    auto reverse_checker = TEST_PREPARE_CALLABLE(std::ranges::shift_left);

    const int test_sz = (1<<18) + 953; // 256K+
    test_range_algo<0>{test_sz}(dpl_ranges::shift_left, reverse_checker, test_sz/4);
    test_range_algo<1>{test_sz}(dpl_ranges::shift_left, reverse_checker, 3 * test_sz/4);
    test_range_algo<2>{small_size}(dpl_ranges::shift_left, reverse_checker, small_size + 2);
    test_range_algo<3>{small_size}(dpl_ranges::shift_left, reverse_checker, 0);
    test_range_algo<4>{small_size}(dpl_ranges::shift_left, reverse_checker, -2);
#endif //_ENABLE_STD_RANGES_TESTING

    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
