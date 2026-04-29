// -*- C++ -*-
//===-- scan_int32_dump_scratch.pass.cpp -----------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Isolation test: Run int32_t inclusive_scan at n=20000 (RTS path) with
// _ONEDPL_RTS_DUMP_SCRATCH enabled. This prints all scratch buffer values
// after the reduce kernel completes, before the scan kernel runs.
//
// The output shows the sub-group partial reductions and carry values that
// the scan kernel will read. Inspect these to verify they are reasonable
// prefix-sum partial results for the given input.

#define _ONEDPL_RTS_DUMP_SCRATCH 1
#define _ONEDPL_REDUCE_THEN_SCAN_DEBUG 1

#include "support/test_config.h"

#include _PSTL_TEST_HEADER(execution)
#include _PSTL_TEST_HEADER(numeric)

#include "support/utils.h"

using namespace TestUtils;

int
main()
{
    using T = std::int32_t;
    T trash = -666;
    auto convert = [](std::uint32_t k) { return T((k % 991 + 1) ^ (k % 997 + 2)); };

    constexpr size_t n = 20000;
    Sequence<T> in(n, convert);
    Sequence<T> out(n, [&](std::int32_t) { return trash; });

    auto policy = oneapi::dpl::execution::dpcpp_default;
    std::inclusive_scan(policy, in.begin(), in.end(), out.begin());

    return done();
}
