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
              typename Pred, typename Proj = std::identity>
    auto operator()(InRange&& r_in, OutRange&& r_out, Pred pred, Proj proj = {})
    {
        using ret_type = std::ranges::remove_copy_if_result<std::ranges::borrowed_iterator_t<InRange>,
                                                            std::ranges::borrowed_iterator_t<OutRange>>;
        auto in = std::ranges::begin(r_in);
        auto out = std::ranges::begin(r_out);
        std::size_t i = 0, j = 0;
        for(; i < std::ranges::size(r_in); ++i)
        {
             if (!std::invoke(pred, std::invoke(proj, in[i])))
             {
                 if (j < std::ranges::size(r_out))
                     out[j++] = in[i];
                 else
                     break;
             }
        }
        return ret_type{in + i, out + j};
    }
} remove_copy_if_checker;
#endif // _ENABLE_STD_RANGES_TESTING

std::int32_t
main()
{
#if _ENABLE_STD_RANGES_TESTING
    using namespace test_std_ranges;
    namespace dpl_ranges = oneapi::dpl::ranges;

    // input generator with a fair chance of repeating the previous value
    auto repeat_sometimes = [](auto i) {
        static decltype(i) last = 0;
        if (i == 0)
            last = 0; // reset
        else if (i%7 > 0 && (last + i - 1)%3 == 0)
            last = i;
        return last;
    };
    using repeating_gen = decltype(repeat_sometimes);
    auto modulo_3_is_1 = [](int val) { return (val % 3) == 1; };

    test_range_algo<0, int, data_in_out_lim>{239}(dpl_ranges::remove_copy_if, remove_copy_if_checker, pred);
    test_range_algo<1, int, data_in_out_lim>{1471}(dpl_ranges::remove_copy_if, remove_copy_if_checker, select_many);
    test_range_algo<2, int, data_in_out_lim>{}(dpl_ranges::remove_copy_if, remove_copy_if_checker, select_many, proj);
    test_range_algo<3, P2, data_in_out_lim, repeating_gen>{}(dpl_ranges::remove_copy_if, remove_copy_if_checker, modulo_3_is_1, &P2::x);
    test_range_algo<4, P2, data_in_out_lim>{}(dpl_ranges::remove_copy_if, remove_copy_if_checker, pred, &P2::proj);
    test_range_algo<5, int, data_in_out_lim>{big_sz}(dpl_ranges::remove_copy_if, remove_copy_if_checker, pred);
    test_range_algo<6, int, data_in_out_lim, repeating_gen>{big_sz}(dpl_ranges::remove_copy_if, remove_copy_if_checker, select_many);
#endif // _ENABLE_STD_RANGES_TESTING

    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
