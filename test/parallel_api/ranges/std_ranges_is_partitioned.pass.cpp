// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "std_ranges_test.h"

std::int32_t
main()
{
#if _ENABLE_STD_RANGES_TESTING
    using namespace test_std_ranges;
    namespace dpl_ranges = oneapi::dpl::ranges;

    auto is_partitioned_checker = TEST_PREPARE_CALLABLE(std::ranges::is_partitioned);

    test_range_algo<0>{big_sz}(dpl_ranges::is_partitioned, is_partitioned_checker, pred1);
    test_range_algo<1>{}(dpl_ranges::is_partitioned, is_partitioned_checker, pred1, proj);
    test_range_algo<2, P2>{}(dpl_ranges::is_partitioned, is_partitioned_checker, pred1, &P2::x);
    test_range_algo<3, P2>{}(dpl_ranges::is_partitioned, is_partitioned_checker, pred1, &P2::proj);
#endif //_ENABLE_STD_RANGES_TESTING

    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
