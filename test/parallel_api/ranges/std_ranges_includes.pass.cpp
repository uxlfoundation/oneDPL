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

    auto includes_checker = TEST_PREPARE_CALLABLE(std::ranges::includes);

    test_range_algo<0, int, data_in_in>{big_sz}(dpl_ranges::includes, includes_checker);
    test_range_algo<1, int, data_in_in>{}(dpl_ranges::includes, includes_checker, std::ranges::less{});

    test_range_algo<2, int, data_in_in>{}(dpl_ranges::includes, includes_checker, std::ranges::less{}, proj);

    // Check with different projections,
    // but when includes returns true - to make sure that the projections are applied correctly.
    // The first sequence is [0, 3, 6, ...], the second is [0, 1, 2, ...],
    // but the second is transformed to [0, 3, 6, ...] by its projection.
    auto x1 = [](auto&& v) { return v; };
    auto x3 = [](auto&& v) { return v * 3; };
    test_range_algo<3, int, data_in_in, decltype(x3), decltype(x1)>{}(
        dpl_ranges::includes, includes_checker, std::ranges::less{}, x1, x3);

    // test_range_algo<4, P2, data_in_in>{}(dpl_ranges::includes, includes_checker, std::ranges::less{}, &P2::x, &P2::x);
    // test_range_algo<5, P2, data_in_in>{}(dpl_ranges::includes, includes_checker, std::ranges::less{}, &P2::proj, &P2::proj);
#endif //_ENABLE_STD_RANGES_TESTING

    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
