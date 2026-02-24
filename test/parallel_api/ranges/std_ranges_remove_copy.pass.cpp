// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) UXL Foundation Contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "std_ranges_test.h"

#if _ENABLE_STD_RANGES_TESTING
struct
{
    template <std::ranges::random_access_range InRange, std::ranges::random_access_range OutRange,
              typename V, typename Proj = std::identity>
    auto operator()(InRange&& r_in, OutRange&& r_out, const V& value, Proj proj = {})
    {
        using ret_type = std::ranges::remove_copy_result<std::ranges::borrowed_iterator_t<InRange>,
                                                         std::ranges::borrowed_iterator_t<OutRange>>;
        auto in = std::ranges::begin(r_in);
        auto out = std::ranges::begin(r_out);
        std::size_t i = 0, j = 0;
        for(; i < std::ranges::size(r_in); ++i)
        {
             if (!std::ranges::equal_to{}(std::invoke(proj, in[i]), value))
             {
                 if (j < std::ranges::size(r_out))
                     out[j++] = in[i];
                 else
                     break;
             }
        }
        return ret_type{in + i, out + j};
    }
} remove_copy_checker;
#endif // _ENABLE_STD_RANGES_TESTING

std::int32_t
main()
{
#if _ENABLE_STD_RANGES_TESTING
    using namespace test_std_ranges;
    namespace dpl_ranges = oneapi::dpl::ranges;

    // input generator with a fair chance of repeating the previous value
    auto almost_always_two = [](auto i) {
        if (i%7 > 0 && (i - 1)%3 == 0)
            return i;
        return 2;
    };
    using many_twos = decltype(almost_always_two);

    test_range_algo<0, int, data_in_out_lim>{179}(dpl_ranges::remove_copy, remove_copy_checker, 0);
    test_range_algo<1, int, data_in_out_lim, many_twos>{1127}(dpl_ranges::remove_copy, remove_copy_checker, 2);
    test_range_algo<2, int, data_in_out_lim>{}(dpl_ranges::remove_copy, remove_copy_checker, 1, proj);
    test_range_algo<3, P2, data_in_out_lim, many_twos>{}(dpl_ranges::remove_copy, remove_copy_checker, 2, &P2::x);
    test_range_algo<4, P2, data_in_out_lim>{}(dpl_ranges::remove_copy, remove_copy_checker, 0, &P2::proj);
    test_range_algo<5, int, data_in_out_lim>{big_sz}(dpl_ranges::remove_copy, remove_copy_checker, 1);
    test_range_algo<6, int, data_in_out_lim, many_twos>{big_sz}(dpl_ranges::remove_copy, remove_copy_checker, 2);
#endif // _ENABLE_STD_RANGES_TESTING

    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
