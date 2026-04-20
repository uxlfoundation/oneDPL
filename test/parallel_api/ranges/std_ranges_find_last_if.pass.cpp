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

#if _ENABLE_STD_RANGES_TESTING
#if __cpp_lib_ranges_find_last >= 202207L
    auto checker = TEST_PREPARE_CALLABLE(std::ranges::find_last_if);
#else
    struct {
        template<std::ranges::forward_range R, typename Pred, typename Proj = std::identity>
        std::ranges::borrowed_subrange_t<R> operator()(R&& r, Pred pred, Proj proj = {})
        {
            std::ranges::iterator_t<R> res{}, it{};
            bool found = false;
            for (it = std::ranges::begin(r); it != std::ranges::end(r); ++it)
            {
                if (bool(std::invoke(pred, std::invoke(proj, *it))))
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

    test_range_algo<0>{big_sz}(dpl_ranges::find_last_if, checker, pred);
    test_range_algo<1>{}(dpl_ranges::find_last_if, checker, pred, proj);
    test_range_algo<2, P2>{}(dpl_ranges::find_last_if, checker, pred3, &P2::x); // not found
    test_range_algo<3, P2>{}(dpl_ranges::find_last_if, checker, pred, &P2::proj);
#endif //_ENABLE_STD_RANGES_TESTING

    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
