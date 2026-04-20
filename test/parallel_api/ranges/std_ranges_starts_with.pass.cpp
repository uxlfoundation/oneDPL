// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2026 UXL Foundation Contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "std_ranges_test.h"

#if _ENABLE_STD_RANGES_TESTING
    template<int call_id, typename T, typename DataGen2 = std::identity>
    using launcher = test_std_ranges::test_range_algo<call_id, T, test_std_ranges::data_in_in,
                                                      /*DataGen1*/ std::identity, DataGen2>;
#if __cpp_lib_ranges_starts_ends_with >= 202106L
    auto checker = TEST_PREPARE_CALLABLE(std::ranges::starts_with);
#else
    struct {
        template<std::ranges::input_range R1, std::ranges::input_range R2, typename Pred = std::ranges::equal_to,
                 typename Proj1 = std::identity, typename Proj2 = std::identity>
        bool operator()(R1&& r1, R2&& r2, Pred pred = {}, Proj1 proj1 = {}, Proj2 proj2 = {})
        {
            auto last = std::ranges::end(r2);
            return std::ranges::mismatch(r1, r2, pred, proj1, proj2).in2 == last;
        }
    } checker;
#endif
#endif

std::int32_t
main()
{
#if _ENABLE_STD_RANGES_TESTING
    using namespace test_std_ranges;
    namespace dpl_ranges = oneapi::dpl::ranges;

    auto almost_always_i = [](auto i){ return (i == medium_size/2 + 19)? 0 : i; };
    using data_gen_needle = decltype(almost_always_i);

    launcher<0, int>{big_sz}(dpl_ranges::starts_with, checker, binary_pred_const);
    launcher<1, int>{}(dpl_ranges::starts_with, checker, binary_pred, proj);
    launcher<2, int, decltype(proj)>{}(dpl_ranges::starts_with, checker, binary_pred, proj);
    launcher<3, P2>{}(dpl_ranges::starts_with, checker, binary_pred_const, &P2::x, &P2::proj);
    launcher<4, P2>{}(dpl_ranges::starts_with, checker, binary_pred, &P2::proj, &P2::x);
    launcher<5, int, data_gen_needle>{}(dpl_ranges::starts_with, checker);
#endif //_ENABLE_STD_RANGES_TESTING

    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
