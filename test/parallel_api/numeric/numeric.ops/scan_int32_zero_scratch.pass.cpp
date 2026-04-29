// -*- C++ -*-
//===-- scan_int32_zero_scratch.pass.cpp -----------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Isolation test: Identical to scan_int32_no_single_group but with
// _ONEDPL_RTS_ZERO_SCRATCH_BETWEEN_KERNELS defined. This zeros the scratch
// buffer between the reduce and scan kernels.
//
// If this PASSES while scan_int32_no_single_group CRASHES:
//   The scan kernel crash depends on corrupt values written by the reduce kernel.
//   Root cause is likely a reduce kernel out-of-bounds write or bad reduction value.
//
// If this CRASHES the same way:
//   The scan kernel crashes regardless of scratch buffer contents, meaning the
//   issue is in the scan kernel's own execution (stack overflow, JIT miscompile, etc.)
//
// NOTE: Results will be numerically wrong since the reduce output is zeroed,
// but we only care about whether it crashes, not correctness.

#define _ONEDPL_RTS_ZERO_SCRATCH_BETWEEN_KERNELS 1

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

    // Just run inclusive_scan — we don't check correctness, only whether it crashes.
    auto policy = oneapi::dpl::execution::dpcpp_default;
    std::inclusive_scan(policy, in.begin(), in.end(), out.begin());

    // If we get here without crashing, the test passes.
    return done();
}
