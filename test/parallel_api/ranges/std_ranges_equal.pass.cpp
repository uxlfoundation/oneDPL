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

    auto equal_checker = TEST_PREPARE_CALLABLE(std::ranges::equal);

    test_range_algo<0, int, data_in_in>{big_sz}(dpl_ranges::equal, equal_checker, binary_pred);
    test_range_algo<1, int, data_in_in>{}(dpl_ranges::equal, equal_checker, binary_pred, proj);
    test_range_algo<2, P2, data_in_in>{}(dpl_ranges::equal, equal_checker, binary_pred, &P2::x, &P2::x);
    test_range_algo<3, P2, data_in_in>{}(dpl_ranges::equal, equal_checker, binary_pred, &P2::proj, &P2::proj);

    // The predicate returns a by-value result referring into the temporaries created by by-value projections.
    // lifetime_checked is host only.
    test_range_algo<4, int, data_in_in>{}.test_range_algo_impl_host(dpl_ranges::equal, equal_checker, equal_ref, proj_to_checked, proj_to_checked);
    check_no_dead_reads("equal read a projected value after its destruction");
#endif //_ENABLE_STD_RANGES_TESTING

    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
