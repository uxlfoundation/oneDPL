// -*- C++ -*-
//===-- scan_int32_noop_kernel.pass.cpp ------------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Isolation test: Submits the full RTS reduce and scan kernels with complete
// template instantiation and *this capture, but work-items return immediately
// without executing the kernel body.
//
// This tests whether the crash occurs during JIT compilation of the kernel
// (template instantiation, SPIR-V generation, native code emission) vs during
// actual kernel execution.
//
// If this CRASHES:
//   The JIT compiler itself is the problem — compiling the deeply-templated
//   kernel body causes the stack buffer overrun during codegen.
//
// If this PASSES:
//   The crash is in the generated code at runtime — actual stack frame usage
//   during kernel execution exceeds what the CPU runtime allocates.

#define _ONEDPL_RTS_NOOP_KERNEL_BODY 1

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

    // Run inclusive_scan — kernels are submitted and JIT'd but bodies are no-ops.
    // We don't check correctness, only whether it crashes.
    auto policy = oneapi::dpl::execution::dpcpp_default;
    std::inclusive_scan(policy, in.begin(), in.end(), out.begin());

    return done();
}
