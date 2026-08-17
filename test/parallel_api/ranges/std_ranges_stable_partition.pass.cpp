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

    auto stable_partition_checker = TEST_PREPARE_CALLABLE(std::ranges::stable_partition);

    test_range_algo<0>{get_scan_big_sz()}(dpl_ranges::stable_partition, stable_partition_checker, pred);
    test_range_algo<1>{}(dpl_ranges::stable_partition, stable_partition_checker, pred, proj);

#if TEST_LONG_RUN
    test_range_algo<2, P2>{}(dpl_ranges::stable_partition, stable_partition_checker, pred, &P2::x);
    test_range_algo<3, P2>{}(dpl_ranges::stable_partition, stable_partition_checker, pred, &P2::proj);
#endif
#endif //_ENABLE_STD_RANGES_TESTING

    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
