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

#include "support/test_config.h"
#include "support/test_macros.h"
#include "support/utils.h"

#if _ENABLE_STD_RANGES_TESTING
#include <ranges>
#include "xpu_std_ranges_test.h"
#endif //_ENABLE_STD_RANGES_TESTING

int
main()
{
#if _ENABLE_STD_RANGES_TESTING
    auto test = [](){
        auto v = std::ranges::views::iota(0, 4);
        auto res = std::ranges::subrange(v.begin() + 1, v.end());

        return res.size() == 3 && res[0] == 1 && res[1] == 2 && res[2] == 3 && (*res.begin() + 2) == 3 && 
            res.end() - res.begin() == 3 && *std::ranges::next(res.begin()) == 2 && *std::ranges::prev(res.end()) == 3;
    };
    const bool res = kernel_test<class std_reverse_test>(test);
    EXPECT_TRUE(res, "Wrong result of subrange check within a kernel");
#endif //_ENABLE_STD_RANGES_TESTING

    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
