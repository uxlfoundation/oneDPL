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
        int a[4] = {0, 0, 0, 0};
        auto res = std::ranges::subrange(a, a+4) | std::ranges::views::transform([](auto v) { return v + 1;});
        return res[0] == 1 && res[1] == 1 && res[2] == 1 && res[3] == 1 && *(res.begin() + 2) == 1 &&
               res.end() - res.begin() == 4;
    };
    const bool res = kernel_test<class std_transform_test>(test);
    EXPECT_TRUE(res, "Wrong result of transform_view check within a kernel");
#endif //_ENABLE_STD_RANGES_TESTING

    return TestUtils::done(_ENABLE_STD_RANGES_TESTING);
}
