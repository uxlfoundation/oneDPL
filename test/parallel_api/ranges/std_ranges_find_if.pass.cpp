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

    auto find_if_checker = TEST_PREPARE_CALLABLE(std::ranges::find_if);

    test_range_algo<0>{big_sz}(dpl_ranges::find_if, find_if_checker, pred);
    test_range_algo<1>{}(dpl_ranges::find_if, find_if_checker, pred, proj);
    test_range_algo<2, P2>{}(dpl_ranges::find_if, find_if_checker, pred, &P2::x);
    test_range_algo<3, P2>{}(dpl_ranges::find_if, find_if_checker, pred, &P2::proj);

    // The predicate returns a reference into the temporary created by a by-value projection.
    // lifetime_checked is host only; the device-friendly case below uses a trivially copyable projected value.
    test_range_algo<4>{}.test_range_algo_impl_host(dpl_ranges::find_if, find_if_checker, &proj_result::flag, proj_to_result);
    // The predicate returns a by-value result referring into the temporary created by a by-value projection.
    test_range_algo<4>{}.test_range_algo_impl_host(dpl_ranges::find_if, find_if_checker, flag_ref, proj_to_result);
    test_range_algo<4>{}(dpl_ranges::find_if, find_if_checker, pred_ref, proj_to_p2);
    check_no_dead_reads("find_if read a projected value after its destruction");
#endif //_ENABLE_STD_RANGES_TESTING

    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
