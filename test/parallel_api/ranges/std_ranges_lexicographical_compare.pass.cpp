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
#endif

std::int32_t
main()
{
#if _ENABLE_STD_RANGES_TESTING
    using namespace test_std_ranges;
    namespace dpl_ranges = oneapi::dpl::ranges;

    auto checker = TEST_PREPARE_CALLABLE(std::ranges::lexicographical_compare);
    auto almost_always_i = [](auto i){ return (i == medium_size/2 + 19)? 0 : i; };
    using data_gen_needle = decltype(almost_always_i);

    launcher<0, int>{big_sz}(dpl_ranges::lexicographical_compare, checker, binary_pred_const);
    launcher<1, int>{}(dpl_ranges::lexicographical_compare, checker, binary_pred, proj);
    launcher<2, P2>{}(dpl_ranges::lexicographical_compare, checker, binary_pred, &P2::x, &P2::proj);
    launcher<3, P2>{}(dpl_ranges::lexicographical_compare, checker, binary_pred, &P2::proj, &P2::x);
    launcher<4, int, decltype(proj)>{}(dpl_ranges::lexicographical_compare, checker, binary_pred, proj);
    launcher<5, int, data_gen_needle>{}(dpl_ranges::lexicographical_compare, checker);
#endif //_ENABLE_STD_RANGES_TESTING

    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
