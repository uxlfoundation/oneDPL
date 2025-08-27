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

    auto reverse_copy_checker = [](std::ranges::random_access_range auto&& r_in,
                                   std::ranges::random_access_range auto&& r_out)
    {
        const auto in_r_size = std::ranges::size(r_in);
        const auto out_r_size = std::ranges::size(r_out);
        const auto sz = std::ranges::min(in_r_size, out_r_size);

        const auto in_r_skipped = in_r_size - sz;

        std::ranges::reverse_copy(
            std::ranges::drop_view(r_in, in_r_skipped),
            std::ranges::take_view(r_out, sz).begin());

        using ret_type = std::ranges::in_in_out_result<std::ranges::borrowed_iterator_t<decltype(r_in)>,
                                                       std::ranges::borrowed_iterator_t<decltype(r_in)>,
                                                       std::ranges::borrowed_iterator_t<decltype(r_out)>>;

        auto last_in = std::ranges::begin(r_in) + in_r_size;
        auto stop_in = std::ranges::begin(r_in) + in_r_skipped;
        auto stop_out = std::ranges::begin(r_out) + sz;
        return ret_type{last_in, stop_in, stop_out};
    };

    test_range_algo<0, int, data_in_out_lim>{big_sz}(dpl_ranges::reverse_copy, reverse_copy_checker);
#endif //_ENABLE_STD_RANGES_TESTING

    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
