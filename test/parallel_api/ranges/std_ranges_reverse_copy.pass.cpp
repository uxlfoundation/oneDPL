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
        const auto in_size = std::ranges::size(r_in);
        const auto out_size = std::ranges::size(r_out);
        const auto skipped = in_size - std::ranges::min(in_size, out_size);

        auto res = std::ranges::reverse_copy(std::ranges::drop_view(r_in, skipped), std::ranges::begin(r_out));

        using ret_type = std::ranges::in_in_out_result<std::ranges::borrowed_iterator_t<decltype(r_in)>,
                                                       std::ranges::borrowed_iterator_t<decltype(r_in)>,
                                                       std::ranges::borrowed_iterator_t<decltype(r_out)>>;

        return ret_type{res.in, std::ranges::begin(r_in) + skipped, res.out};
    };

    test_range_algo<0, int, data_in_out_lim>{big_sz}(dpl_ranges::reverse_copy, reverse_copy_checker);
#endif //_ENABLE_STD_RANGES_TESTING

    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
