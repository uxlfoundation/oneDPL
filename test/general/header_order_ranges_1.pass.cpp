// -*- C++ -*-
//===-- header_inclusion_order_algorithm_0.pass.cpp -----------------------===//
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

#if _ENABLE_RANGES_TESTING
#include _PSTL_TEST_HEADER(ranges)
#include _PSTL_TEST_HEADER(execution)
#endif

#include "support/utils.h"

int
main()
{
#if _ENABLE_RANGES_TESTING
    using namespace oneapi::dpl::experimental::ranges;
    all_of(TestUtils::default_dpcpp_policy, views::fill(-1, 10), [](auto i) { return i == -1;});
#endif

    return TestUtils::done(_ENABLE_RANGES_TESTING);
}
