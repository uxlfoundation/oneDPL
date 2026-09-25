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
struct
{
    template <std::ranges::random_access_range InRange, std::ranges::random_access_range OutRange,
              typename Pred, typename Proj = std::identity>
    auto operator()(InRange&& r_in, OutRange&& r_out, Pred pred, Proj proj = {})
    {
        using ret_type = std::ranges::copy_if_result<std::ranges::borrowed_iterator_t<InRange>,
                                                     std::ranges::borrowed_iterator_t<OutRange>>;
        auto in = std::ranges::begin(r_in);
        auto out = std::ranges::begin(r_out);
        std::size_t i = 0, j = 0;
        for(; i < std::ranges::size(r_in); ++i)
        {
             if (std::invoke(pred, std::invoke(proj, in[i])))
             {
                 if (j < std::ranges::size(r_out))
                     out[j++] = in[i];
                 else
                     break;
             }
        }
        return ret_type{in + i, out + j};
    }
} copy_if_checker;
#endif // _ENABLE_STD_RANGES_TESTING

std::int32_t
main()
{
#if _ENABLE_STD_RANGES_TESTING
    using namespace test_std_ranges;
    namespace dpl_ranges = oneapi::dpl::ranges;

    test_range_algo<0, int, data_in_out_lim>{217}(dpl_ranges::copy_if, copy_if_checker, pred);
    test_range_algo<1, int, data_in_out_lim>{1234}(dpl_ranges::copy_if, copy_if_checker, select_many);
    test_range_algo<2, int, data_in_out_lim>{}(dpl_ranges::copy_if, copy_if_checker, select_many, proj);
    test_range_algo<3, P2, data_in_out_lim>{}(dpl_ranges::copy_if, copy_if_checker, pred, &P2::x);
    test_range_algo<4, P2, data_in_out_lim>{}(dpl_ranges::copy_if, copy_if_checker, pred, &P2::proj);
    test_range_algo<5, int, data_in_out_lim>{get_scan_big_sz()}(dpl_ranges::copy_if, copy_if_checker, pred);
    test_range_algo<6, int, data_in_out_lim>{get_scan_big_sz()}(dpl_ranges::copy_if, copy_if_checker, select_many);

    // The predicate returns a reference into the temporary created by a by-value projection.
    // lifetime_checked is host only; the device-friendly case below uses a trivially copyable projected value.
    test_range_algo<7, int, data_in_out_lim>{}.test_range_algo_impl_host(dpl_ranges::copy_if, copy_if_checker, &proj_result::flag, proj_to_result);
    // The predicate returns a by-value result referring into the temporary created by a by-value projection.
    test_range_algo<7, int, data_in_out_lim>{}.test_range_algo_impl_host(dpl_ranges::copy_if, copy_if_checker, flag_ref, proj_to_result);
    test_range_algo<7, int, data_in_out_lim>{}(dpl_ranges::copy_if, copy_if_checker, pred_ref, proj_to_p2);
    check_no_dead_reads("copy_if read a projected value after its destruction");
#endif // _ENABLE_STD_RANGES_TESTING

    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
