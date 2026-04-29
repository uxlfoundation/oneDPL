// -*- C++ -*-
//===-- scan_int32_reduce_only.pass.cpp -----------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Isolation test: Skip the scan kernel, only run the reduce kernel.
// The output will be garbage, but we're testing whether the crash comes
// from the reduce kernel or the scan kernel.
// If this passes: the scan kernel is the crash source.
// If this crashes: the reduce kernel is the crash source.
// NOTE: correctness checking is disabled since we skip the scan step.

#define _ONEDPL_RTS_SKIP_SCAN
#define _ONEDPL_REDUCE_THEN_SCAN_DEBUG 0

#include "support/test_config.h"

#include _PSTL_TEST_HEADER(execution)
#include _PSTL_TEST_HEADER(numeric)

#include "support/utils.h"
#include "support/scan_serial_impl.h"

using namespace TestUtils;

struct test_inclusive_no_check
{
    template <typename Policy, typename Iterator1, typename Iterator2, typename Iterator3, typename Size, typename T>
    std::enable_if_t<!TestUtils::is_reverse_v<Iterator1>>
    operator()(Policy&& exec, Iterator1 in_first, Iterator1 in_last, Iterator2 out_first, Iterator2 out_last,
               Iterator3 expected_first, Iterator3, Size n, T init, T trash)
    {
        // Run inclusive_scan but don't check output — scan kernel is skipped
        auto orr = std::inclusive_scan(std::forward<Policy>(exec), in_first, in_last, out_first);
        (void)orr;
    }
    template <typename Policy, typename Iterator1, typename Iterator2, typename Iterator3, typename Size, typename T>
    std::enable_if_t<TestUtils::is_reverse_v<Iterator1>>
    operator()(Policy&&, Iterator1, Iterator1, Iterator2, Iterator2, Iterator3, Iterator3, Size, T, T) {}
};

int
main()
{
    using T = std::int32_t;
    T init = 0;
    T trash = -666;
    auto convert = [](std::uint32_t k) { return T((k % 991 + 1) ^ (k % 997 + 2)); };

    // n=20000 to ensure RTS path
    constexpr size_t n = 20000;
    Sequence<T> in(n, convert);
    Sequence<T> expected(n);
    Sequence<T> out(n, [&](std::int32_t) { return trash; });

    invoke_on_all_policies<0>()(test_inclusive_no_check(), in.begin(), in.end(), out.begin(),
                                out.end(), expected.begin(), expected.end(), in.size(), init, trash);

    return done();
}
