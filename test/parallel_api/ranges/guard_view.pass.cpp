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

#include <oneapi/dpl/execution>

#if _ENABLE_RANGES_TESTING
#    include <oneapi/dpl/ranges>
#endif

#include "support/utils.h"

#include <iostream>

std::int32_t
main()
{
#if _ENABLE_RANGES_TESTING
    using CountItr = oneapi::dpl::counting_iterator<uint64_t>;
    CountItr count_itr(0UL);

    const size_t max_int32p2 = (size_t)::std::numeric_limits<int32_t>::max() + 2UL;

    oneapi::dpl::__ranges::guard_view<CountItr> gview{count_itr, max_int32p2};

    //check simple access
    for (int i = 0; i < 10; i++)
    {
        EXPECT_TRUE(gview[i] == i, "wrong effect with guard_view");
    }
    const size_t last_idx = gview.size() - 1;
    //check access with index greater than 32 bit integer max
    EXPECT_TRUE(gview[last_idx] == last_idx, "wrong effect with guard_view with index greater than max int32");

#endif //_ENABLE_RANGES_TESTING

    return TestUtils::done(_ENABLE_RANGES_TESTING);
}
