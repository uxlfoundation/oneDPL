// -*- C++ -*-
//===-- header_inclusion_order_algorithm_1.pass.cpp -----------------------===//
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

#include _PSTL_TEST_HEADER(algorithm)
#include _PSTL_TEST_HEADER(execution)

#include <type_traits>

// Check that oneDPL-specific numeric algorithms are still available in 'algorithm'
using pint = int*;
using pfloat = float*;
using pdouble = double*;

static_assert(std::is_same_v<pdouble,
    decltype(oneapi::dpl::exclusive_scan_by_segment(oneapi::dpl::execution::par, pint{}, pint{}, pfloat{}, pdouble{}))>);
static_assert(std::is_same_v<pdouble,
    decltype(oneapi::dpl::inclusive_scan_by_segment(oneapi::dpl::execution::par, pint{}, pint{}, pfloat{}, pdouble{}))>);
static_assert(std::is_same_v<std::pair<pint, pdouble>,
    decltype(oneapi::dpl::reduce_by_segment(oneapi::dpl::execution::par, pint{}, pint{}, pfloat{}, pint{}, pdouble{}))>);
static_assert(std::is_same_v<pint,
    decltype(oneapi::dpl::histogram(oneapi::dpl::execution::par, pdouble{}, pdouble{}, pfloat{}, pfloat{}, pint{}))>);

#include "support/utils.h"

int
main()
{

    return TestUtils::done();
}
