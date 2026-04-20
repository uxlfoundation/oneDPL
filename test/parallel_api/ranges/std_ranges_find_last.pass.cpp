// -*- C++ -*-
//===------------------------------------------------------===//
//
// Copyright (C) UXL Foundation Contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===------------------------------------------------------===//

#include "std_ranges_test.h"

#if _ENABLE_STD_RANGES_TESTING
#if __cpp_lib_ranges_find_last >= 202207L
    auto checker = TEST_PREPARE_CALLABLE(std::ranges::find_last);
#else
    struct {
        template<std::ranges::forward_range R, typename T, typename Proj = std::identity>
        std::ranges::borrowed_subrange_t<R> operator()(R&& r, const T& val, Proj proj = {})
        {
            std::ranges::iterator_t<R> res{}, it{};
            bool found = false;
            for (it = std::ranges::begin(r); it != std::ranges::end(r); ++it)
            {
                if (std::invoke(proj, *it) == val)
                {
                    res = it;
                    found = true;
                }
            }
            return {found? res : it, it};
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

    test_range_algo<0>{big_sz}(dpl_ranges::find_last, checker, 314);
    test_range_algo<1>{}(dpl_ranges::find_last, checker, 271, proj);
    test_range_algo<2, P2>{}(dpl_ranges::find_last, checker, 99, &P2::x);
    test_range_algo<3, P2>{}(dpl_ranges::find_last, checker, -359, &P2::proj); // not found
#endif //_ENABLE_STD_RANGES_TESTING

    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
