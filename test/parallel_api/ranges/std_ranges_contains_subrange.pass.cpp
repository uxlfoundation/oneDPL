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
#if __cpp_lib_ranges_contains >= 202207L
    auto checker = TEST_PREPARE_CALLABLE(std::ranges::contains_subrange);
#else
    struct {
        
        template<std::ranges::forward_range R1, std::ranges::forward_range R2, typename Pred = std::ranges::equal_to,
                 typename Proj1 = std::identity, typename Proj2 = std::identity>
        bool operator()(R1&& r1, R2&& r2, Pred pred = {}, Proj1 proj1 = {}, Proj2 proj2 = {})
        {
            if (std::ranges::begin(r2) == std::ranges::end(r2))
                return true;
            auto&& result = std::ranges::search(r1, r2, pred, proj1, proj2);
            return !result.empty();
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

    auto lam = [](auto i) { return i + 371; };
    using data_gen_shifted = decltype(lam);

    // Sizes of both sequences will vary in the test, so each call might test both successful and unsuccessful searches
    launcher<0, int>{big_sz}(dpl_ranges::contains_subrange, checker, binary_pred_const);
    launcher<1, int>{}(dpl_ranges::contains_subrange, checker);
    launcher<2, int>{}(dpl_ranges::contains_subrange, checker, binary_pred, dpl::identity{});
    launcher<3, int, data_gen_shifted>{big_sz}(dpl_ranges::contains_subrange, checker, binary_pred, proj, proj);
    launcher<4, P3, data_gen_shifted>{}(dpl_ranges::contains_subrange, checker, binary_pred_const, &P3::x, &P3::proj);
    launcher<5, P3>{}(dpl_ranges::contains_subrange, checker, std::equal_to<>{}, &P3::proj, &P3::y);
#endif //_ENABLE_STD_RANGES_TESTING

    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
