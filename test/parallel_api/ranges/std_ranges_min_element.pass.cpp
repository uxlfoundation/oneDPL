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

    auto min_element_checker = TEST_PREPARE_CALLABLE(std::ranges::min_element);

    test_range_algo<0>{big_sz}(dpl_ranges::min_element, min_element_checker, std::ranges::less{});

    test_range_algo<1>{}(dpl_ranges::min_element, min_element_checker, std::ranges::less{}, proj);
    test_range_algo<2, P2>{}(dpl_ranges::min_element, min_element_checker, std::ranges::less{}, &P2::x);
    test_range_algo<3, P2>{}(dpl_ranges::min_element, min_element_checker, std::ranges::less{}, &P2::proj);

    test_range_algo<4>{}(dpl_ranges::min_element, min_element_checker, std::ranges::greater{}, proj);
    test_range_algo<5, P2>{}(dpl_ranges::min_element, min_element_checker, std::ranges::greater{}, &P2::x);
    test_range_algo<6, P2>{}(dpl_ranges::min_element, min_element_checker, std::ranges::greater{}, &P2::proj);

    test_range_algo<7>{}(dpl_ranges::min_element, min_element_checker);
#endif //_ENABLE_STD_RANGES_TESTING

    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
