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
//A checker below modifies a return type; a range based version with policy has another return type.
struct merge_checker_fn
{
    template<std::ranges::random_access_range _R1, std::ranges::random_access_range _R2, std::ranges::random_access_range _ROut,
    typename Comp  = std::ranges::less, typename Proj1 = std::identity, typename Proj2 = std::identity>
    auto operator()(_R1&& r_1, _R2&& r_2, _ROut&& r_out, Comp comp = {}, Proj1 proj1 = {}, Proj2 proj2 = {})
    {
        using ret_type = std::ranges::merge_result<std::ranges::borrowed_iterator_t<decltype(r_1)>,
            std::ranges::borrowed_iterator_t<decltype(r_2)>, std::ranges::borrowed_iterator_t<decltype(r_out)>>;

        auto it_out = std::ranges::begin(r_out);
        auto it_1 = std::ranges::begin(r_1);
        auto it_2 = std::ranges::begin(r_2);
        auto it_1_e = std::ranges::end(r_1);
        auto it_2_e = std::ranges::end(r_2);
        auto it_out_e = std::ranges::end(r_out);

        while(it_1 != it_1_e && it_2 != it_2_e)
        {
            if (std::invoke(comp, std::invoke(proj2, *it_2), std::invoke(proj1, *it_1)))
            {
                *it_out = *it_2;
                ++it_out, ++it_2;
            }
            else
            {
                *it_out = *it_1;
                ++it_out, ++it_1;
            }
            if(it_out == it_out_e)
                return ret_type{it_1, it_2, it_out};
        }
    
        if(it_1 == it_1_e)
        {
            for(; it_2 != it_2_e && it_out != it_out_e; ++it_2, ++it_out)
                *it_out = *it_2;
        }
        else
        {
            for(; it_1 != it_1_e && it_out != it_out_e; ++it_1, ++it_out)
                *it_out = *it_1;
        }

        return ret_type{it_1, it_2, it_out};
    }
} merge_checker;
#endif //_ENABLE_STD_RANGES_TESTING

std::int32_t
main()
{
#if _ENABLE_STD_RANGES_TESTING
    using namespace test_std_ranges;
    namespace dpl_ranges = oneapi::dpl::ranges;

    test_range_algo<0, int, data_in_in_out_lim>{big_sz}(dpl_ranges::merge, merge_checker, std::ranges::less{}, std::identity{}, std::identity{});

    test_range_algo<1, int, data_in_in_out_lim>{}(dpl_ranges::merge, merge_checker, std::ranges::less{}, proj, proj);
    test_range_algo<2, P2, data_in_in_out_lim>{}(dpl_ranges::merge, merge_checker, std::ranges::less{}, &P2::x, &P2::x);
    test_range_algo<3, P2, data_in_in_out_lim>{}(dpl_ranges::merge, merge_checker, std::ranges::less{}, &P2::proj, &P2::proj);

    test_range_algo<4, int, data_in_in_out_lim>{}(dpl_ranges::merge, merge_checker, std::ranges::greater{}, proj, proj);
    test_range_algo<5, P2, data_in_in_out_lim>{}(dpl_ranges::merge, merge_checker, std::ranges::greater{}, &P2::x, &P2::x);
    test_range_algo<6, P2, data_in_in_out_lim>{}(dpl_ranges::merge, merge_checker, std::ranges::greater{}, &P2::proj, &P2::proj);

    test_range_algo<7, int, data_in_in_out_lim>{}(dpl_ranges::merge, merge_checker);
#endif //_ENABLE_STD_RANGES_TESTING

    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
