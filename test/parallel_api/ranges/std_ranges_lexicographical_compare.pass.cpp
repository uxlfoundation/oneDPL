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
    auto plus_one = [](auto i){ return i + 1; };
    auto almost_always_i = [](auto i){ return (i == medium_size/2 + 19)? 0 : i; };
    using data_gen_needle = decltype(almost_always_i);

    launcher<0, int>{big_sz}(dpl_ranges::lexicographical_compare, checker);
    launcher<1, int>{}(dpl_ranges::lexicographical_compare, checker, std::ranges::greater{}, plus_one);
    launcher<2, P2>{}(dpl_ranges::lexicographical_compare, checker, std::ranges::greater{}, &P2::x, &P2::proj);
    launcher<3, P2>{}(dpl_ranges::lexicographical_compare, checker, std::ranges::less{}, &P2::proj, &P2::x);
    launcher<4, int, decltype(plus_one)>{}(dpl_ranges::lexicographical_compare, checker, std::ranges::less{}, plus_one);
    launcher<5, int, data_gen_needle>{}(dpl_ranges::lexicographical_compare, checker);
    launcher<6, int, data_gen_needle>{}(dpl_ranges::lexicographical_compare, checker, std::ranges::greater{});

#if !_TEST_CPP20_RANGES_BROKEN_WRONG_RESULT_LEXICOGRAPHICAL_COMPARE_PROJ1_HOST
    auto gen_hidden_pair_1 = [](auto i) { return (i == 0)? 5 : ((i == 1)? 1 : 0); };
    auto gen_hidden_pair_2 = [](auto i) { return (i == 0)? 4 : ((i == 1)? 9 : 0); };
    test_range_algo<7, int, data_in_in, decltype(gen_hidden_pair_1), decltype(gen_hidden_pair_2)>{}(
        dpl_ranges::lexicographical_compare, checker, std::ranges::less{}, proj);
#endif // !_TEST_CPP20_RANGES_BROKEN_WRONG_RESULT_LEXICOGRAPHICAL_COMPARE_PROJ1_HOST

#if !_TEST_CPP20_RANGES_BROKEN_REQUIRES_LEXICOGRAPHICAL_COMPARE
    check_mixed_types_in_in_host(dpl_ranges::lexicographical_compare, checker, {{1}, {2}, {3}}, {{1}, {2}, {4}},
                                 result_as_is, std::ranges::less{}, proj_a, proj_b);
#if TEST_DPCPP_BACKEND_PRESENT
    check_mixed_types_in_in_device(dpl_ranges::lexicographical_compare, checker, {{1}, {2}, {3}}, {{1}, {2}, {4}},
                                   result_as_is, std::ranges::less{}, proj_a, proj_b);
#endif
#endif // !_TEST_CPP20_RANGES_BROKEN_REQUIRES_LEXICOGRAPHICAL_COMPARE
#endif //_ENABLE_STD_RANGES_TESTING

    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
