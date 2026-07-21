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

#if _ENABLE_STD_RANGES_TESTING && !TEST_DPCPP_BACKEND_PRESENT
struct
{
    template <std::ranges::random_access_range InRange, std::ranges::random_access_range OutRange1,
              std::ranges::random_access_range OutRange2, typename Pred, typename Proj = std::identity>
    auto operator()(InRange&& r_in, OutRange1&& r_true, OutRange2&& r_false, Pred pred, Proj proj = {})
    {
        using ret_type = std::ranges::partition_copy_result<std::ranges::borrowed_iterator_t<InRange>,
                                                            std::ranges::borrowed_iterator_t<OutRange1>,
                                                            std::ranges::borrowed_iterator_t<OutRange2>>;
        auto in = std::ranges::begin(r_in);
        auto out_true = std::ranges::begin(r_true);
        auto out_false = std::ranges::begin(r_false);
        std::size_t i = 0, j = 0, k = 0;
        for(; i < std::ranges::size(r_in); ++i)
        {
             if (std::invoke(pred, std::invoke(proj, in[i])))
             {
                 if (j < std::ranges::size(r_true))
                     out_true[j++] = in[i];
                 else
                     break;
             }
             else
             {
                 if (k < std::ranges::size(r_false))
                     out_false[k++] = in[i];
                 else
                     break;
             }
        }
        return ret_type{in + i, out_true + j, out_false + k};
    }
} partition_copy_checker;
#endif // _ENABLE_STD_RANGES_TESTING

std::int32_t
main()
{
#if _ENABLE_STD_RANGES_TESTING && !TEST_DPCPP_BACKEND_PRESENT
    using namespace test_std_ranges;
    namespace dpl_ranges = oneapi::dpl::ranges;

    test_range_algo<0, int, data_in_out_out_lim>{217}(dpl_ranges::partition_copy, partition_copy_checker, pred);
    test_range_algo<1, int, data_in_out_out_lim>{1234}(dpl_ranges::partition_copy, partition_copy_checker, even_odd);
    test_range_algo<2, int, data_in_out_out_lim>{}(dpl_ranges::partition_copy, partition_copy_checker, select_many, proj);
    test_range_algo<3, P2, data_in_out_out_lim>{}(dpl_ranges::partition_copy, partition_copy_checker, pred, &P2::x);
    test_range_algo<4, P2, data_in_out_out_lim>{}(dpl_ranges::partition_copy, partition_copy_checker, even_odd, &P2::proj);
    test_range_algo<5, int, data_in_out_out_lim>{get_scan_big_sz()}(dpl_ranges::partition_copy, partition_copy_checker, even_odd);
    test_range_algo<6, int, data_in_out_out_lim>{get_scan_big_sz()}(dpl_ranges::partition_copy, partition_copy_checker, select_many);
#endif // _ENABLE_STD_RANGES_TESTING

    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
