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

    auto find_end_checker = TEST_PREPARE_CALLABLE(std::ranges::find_end);

    //true result; data generator is a 'gen(i)', so std::identity produces 0, 1, 2, ...
    test_range_algo<0, int, data_in_in, std::identity/*data generator*/, 
                   std::identity/*data generator*/>{medium_size}(dpl_ranges::find_end, find_end_checker, binary_pred);

    //false result
    test_range_algo<1, int, data_in_in>{big_sz}(dpl_ranges::find_end, find_end_checker, binary_pred);
    test_range_algo<2, int, data_in_in>{}(dpl_ranges::find_end, find_end_checker, binary_pred_const, proj);
    test_range_algo<3, P2, data_in_in>{}(dpl_ranges::find_end, find_end_checker, binary_pred, &P2::x, &P2::x);
    test_range_algo<4, P2, data_in_in>{}(dpl_ranges::find_end, find_end_checker, binary_pred, &P2::proj, &P2::proj);

    // Check with different projections, but when find_end finds the subsequence - to make sure that the
    // projections are applied to the right sequences. The first sequence is [0, 1, 2, ...], the second one
    // is [2, 4, 6, ...], so the subsequence is found (at the offset 1) only if `proj` (v * 2) is applied
    // to the first sequence, which turns it into [0, 2, 4, ...].
    auto gen_2i_2 = [](auto i) { return 2 * i + 2; };
    test_range_algo<5, int, data_in_in, std::identity, decltype(gen_2i_2)>{}(dpl_ranges::find_end, find_end_checker,
                                                                            binary_pred, proj);

    // Check if projections are applied to the right sequences and trigger a compile-time error if not
    check_mixed_types_in_in_host(dpl_ranges::find_end, find_end_checker, {{1}, {2}, {1}, {2}, {5}}, {{1}, {2}},
                                 result_subrange, binary_pred, proj_a, proj_b);
#if TEST_DPCPP_BACKEND_PRESENT
    check_mixed_types_in_in_device(dpl_ranges::find_end, find_end_checker, {{1}, {2}, {1}, {2}, {5}}, {{1}, {2}},
                                   result_subrange, binary_pred, proj_a, proj_b);
#endif
#endif //_ENABLE_STD_RANGES_TESTING

    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
