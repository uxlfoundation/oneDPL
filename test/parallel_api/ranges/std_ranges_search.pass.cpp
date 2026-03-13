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
    template<int call_id, typename T, typename DataGen2 = std::identity>
    using launcher = test_std_ranges::test_range_algo<call_id, T, test_std_ranges::data_in_in,
                                                      /*DataGen1*/ std::identity, DataGen2>;
#endif

std::int32_t
main()
{
#if _ENABLE_STD_RANGES_TESTING
    using namespace test_std_ranges;
    namespace dpl_ranges = oneapi::dpl::ranges;

    auto search_checker = TEST_PREPARE_CALLABLE(std::ranges::search);
    auto lam = [](auto i) { return i + 371; };
    using data_gen_shifted = decltype(lam);

    launcher<0, int>{big_sz}(dpl_ranges::search, search_checker, binary_pred);
    launcher<1, int>{}(dpl_ranges::search, search_checker);
    launcher<2, int>{}(dpl_ranges::search, search_checker, binary_pred_const, proj);
    launcher<3, int, data_gen_shifted>{big_sz}(dpl_ranges::search, search_checker, binary_pred_const, proj, proj);
    launcher<4, P3, data_gen_shifted>{}(dpl_ranges::search, search_checker, binary_pred, &P3::x, &P3::proj);
    launcher<5, P3>{}(dpl_ranges::search, search_checker, std::equal_to<>{}, &P3::proj, &P3::y);
#endif //_ENABLE_STD_RANGES_TESTING

    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
