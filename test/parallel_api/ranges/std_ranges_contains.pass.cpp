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
#if __cpp_lib_ranges_contains >= 202207L
    auto contains_checker = TEST_PREPARE_CALLABLE(std::ranges::contains);
#else
    struct {
        template<std::ranges::input_range R, typename V, typename Proj = std::identity>
        bool operator()(R&& r, V value, Proj proj = {})
        {
            auto last = std::ranges::end(r);
            return std::ranges::find(std::ranges::begin(r), last, value, proj) != last;
        }
    } contains_checker;
#endif
#endif

std::int32_t
main()
{
#if _ENABLE_STD_RANGES_TESTING
    using namespace test_std_ranges;
    namespace dpl_ranges = oneapi::dpl::ranges;

    // expected to be found
    test_range_algo<0>{big_sz}(dpl_ranges::contains, contains_checker, small_size - 19);
    test_range_algo<1>{}(dpl_ranges::contains, contains_checker, proj(small_size/2 + 28), proj);
    test_range_algo<2, P2>{}(dpl_ranges::contains, contains_checker, 137, &P2::x);

    // expected to be absent
    test_range_algo<3, P2>{}(dpl_ranges::contains, contains_checker, -27, &P2::proj);
    test_range_algo<4>{big_sz}(dpl_ranges::contains, contains_checker, -43);
#endif //_ENABLE_STD_RANGES_TESTING

    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
