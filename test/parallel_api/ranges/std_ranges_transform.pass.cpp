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
    auto transform_unary_checker = [](std::ranges::random_access_range auto&& r_in,
                                      std::ranges::random_access_range auto&& r_out, auto&&... args)
    {
        using Size = std::common_type_t<std::ranges::range_size_t<decltype(r_in)>,
            std::ranges::range_size_t<decltype(r_out)>>;
        Size size = std::ranges::min((Size)std::ranges::size(r_out), (Size)std::ranges::size(r_in));

        std::ranges::transform(std::ranges::take_view(r_in, size),
            std::ranges::take_view(r_out, size).begin(), std::forward<decltype(args)>(args)...);

        using ret_type = std::ranges::unary_transform_result<std::ranges::borrowed_iterator_t<decltype(r_in)>,
            std::ranges::borrowed_iterator_t<decltype(r_out)>>;
        return ret_type{std::ranges::begin(r_in) + size, std::ranges::begin(r_out) +  size};
    };

    test_range_algo<0, int, data_in_out_lim>{big_sz}(dpl_ranges::transform, transform_unary_checker, f);
    test_range_algo<1, int, data_in_out_lim>{}(dpl_ranges::transform, transform_unary_checker, f, proj);
    test_range_algo<2, P2, data_in_out_lim>{}(dpl_ranges::transform, transform_unary_checker, f, &P2::x);
    test_range_algo<3, P2, data_in_out_lim>{}(dpl_ranges::transform, transform_unary_checker, f, &P2::proj);

    //A checker below modifies a return type; a range based version with policy has another return type.
    auto transform_binary_checker = [](std::ranges::random_access_range auto&& r_1,
                                       std::ranges::random_access_range auto&& r_2,
                                       std::ranges::random_access_range auto&& r_out, auto&&... args)
    {
        using Size = std::common_type_t<range_size_t<decltype(r_1)>, range_size_t<decltype(r_2)>,
            std::ranges::range_size_t<decltype(r_out)>>;
        Size size = std::ranges::size(r_out);
        if constexpr(std::ranges::sized_range<decltype(r_1)>)
            size = std::ranges::min(size, (Size)std::ranges::size(r_1));
        if constexpr(std::ranges::sized_range<decltype(r_2)>)
            size = std::ranges::min(size, (Size)std::ranges::size(r_2));

        std::ranges::transform(std::ranges::subrange(std::ranges::begin(r_1), std::ranges::begin(r_1) + size),
            std::ranges::subrange(std::ranges::begin(r_2), std::ranges::begin(r_2) + size),
            std::ranges::take_view(r_out, size).begin(), std::forward<decltype(args)>(args)...);

        using ret_type = std::ranges::binary_transform_result<std::ranges::borrowed_iterator_t<decltype(r_1)>,
            std::ranges::borrowed_iterator_t<decltype(r_2)>, std::ranges::borrowed_iterator_t<decltype(r_out)>>;
        return ret_type{std::ranges::begin(r_1) + size, std::ranges::begin(r_2) + size, std::ranges::begin(r_out) + size};
    };

    test_range_algo<4, int, data_in_in_out_lim>{big_sz}(dpl_ranges::transform, transform_binary_checker, binary_f);
    test_range_algo<5, int, data_in_in_out_lim>{}(dpl_ranges::transform, transform_binary_checker, binary_f, proj, proj);
    test_range_algo<6, P2, data_in_in_out_lim>{}(dpl_ranges::transform, transform_binary_checker, binary_f, &P2::x, &P2::x);
    test_range_algo<7, P2, data_in_in_out_lim>{}(dpl_ranges::transform, transform_binary_checker, binary_f, &P2::proj, &P2::proj);
#endif //_ENABLE_STD_RANGES_TESTING

    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
