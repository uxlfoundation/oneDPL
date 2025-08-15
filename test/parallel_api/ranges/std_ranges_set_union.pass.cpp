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

#if 0
    {
        std::vector<int> __r1 = {0, 1, 2, 3, 4};
        std::vector<int> __r2 = {0, 0, 0, 1, 1};
        std::vector<int> __r3_std(__r1.size() + __r2.size());
        std::vector<int> __r3_std_ranges = __r3_std;

        auto __comp  = std::less<int>{};
        auto __proj1 = test_std_ranges::proj;
        auto __proj2 = std::identity{};
        oneapi::dpl::__internal::__binary_op<decltype(__comp), decltype(__proj1), decltype(__proj2)> __comp_2{__comp, __proj1, __proj2};

        std::set_union(__r1.begin(), __r1.end(), __r2.begin(), __r2.end(), __r3_std.begin(), __comp_2);
        std::ranges::set_union(__r1, __r2, std::ranges::begin(__r3_std_ranges), __comp, __proj1, __proj2);

        EXPECT_EQ_N(__r3_std.begin(), __r3_std_ranges.begin(), __r3_std.size(), "");
    }
#endif

    using namespace test_std_ranges;
    namespace dpl_ranges = oneapi::dpl::ranges;

    auto set_union_checker = [](std::ranges::random_access_range auto&& r1,
                                std::ranges::random_access_range auto&& r2,
                                std::ranges::random_access_range auto&& r_out, auto&&... args)
    {
        auto res = std::ranges::set_union(std::forward<decltype(r1)>(r1), std::forward<decltype(r2)>(r2),
                                          std::ranges::begin(r_out), std::forward<decltype(args)>(args)...);

        using ret_type = std::ranges::set_union_result<std::ranges::borrowed_iterator_t<decltype(r1)>,
                                                       std::ranges::borrowed_iterator_t<decltype(r2)>,
                                                       std::ranges::borrowed_iterator_t<decltype(r_out)>>;

        __print_range(std::cout, r_out, "std::ranges::set_union : ");

        return ret_type{res.in1, res.in2, res.out};
    };

    test_range_algo<0, int, data_in_in_out>{big_sz}(dpl_ranges::set_union, set_union_checker);
    test_range_algo<1, int, data_in_in_out>{big_sz}(dpl_ranges::set_union, set_union_checker, std::ranges::less{}, proj);

    test_range_algo<2, int, data_in_in_out>{}(dpl_ranges::set_union, set_union_checker, std::ranges::less{}, proj, proj);
    test_range_algo<3,  P2, data_in_in_out>{}(dpl_ranges::set_union, set_union_checker, std::ranges::less{}, &P2::x, &P2::x);
    test_range_algo<4,  P2, data_in_in_out>{}(dpl_ranges::set_union, set_union_checker, std::ranges::less{}, &P2::proj, &P2::proj);
#endif //_ENABLE_STD_RANGES_TESTING

    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
