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

    //A checker below modifies a return type; a range based version with policy has another return type.
    auto for_each_checker = [](auto&&... args) {
        auto res = std::ranges::for_each(std::forward<decltype(args)>(args)...);
        return res.in;
    };

    test_range_algo<0>{big_sz}(dpl_ranges::for_each, for_each_checker, f_mutuable);
    test_range_algo<1>{}(dpl_ranges::for_each, for_each_checker, f_mutuable, proj_mutuable);
    test_range_algo<2, P2>{}(dpl_ranges::for_each, for_each_checker, f_mutuable, &P2::x);
    test_range_algo<3, P2>{}(dpl_ranges::for_each, for_each_checker, f_mutuable, &P2::proj);

    // The functor returns a reference to a non-copyable object; the projection returns by value.
    // lifetime_checked and the global accumulator are host only.
    auto for_each_acc_checker = [](auto&& r, auto, auto proj) {
        return std::ranges::for_each(std::forward<decltype(r)>(r), sub_from_acc, proj).in;
    };
    test_range_algo<4>{}.test_range_algo_impl_host(dpl_ranges::for_each, for_each_acc_checker, add_to_acc, proj_to_checked);
    EXPECT_EQ(0, for_each_acc.sum.load(), "for_each did not invoke the functor exactly once per element");
    check_no_dead_reads("for_each read a projected value after its destruction");
#endif //_ENABLE_STD_RANGES_TESTING

    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
