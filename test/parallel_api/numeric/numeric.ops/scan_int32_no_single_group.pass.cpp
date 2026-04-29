// -*- C++ -*-
//===-- scan_int32_no_single_group.pass.cpp -------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Isolation test: Run int32_t scan ONLY at n=20000 to ensure the RTS path
// is taken (CPU single-group cutoff is 16384). Confirms RTS crash for int32_t
// and avoids the single-group path which we know works.

#define _ONEDPL_REDUCE_THEN_SCAN_DEBUG 0

#include "support/test_config.h"

#include _PSTL_TEST_HEADER(execution)
#include _PSTL_TEST_HEADER(numeric)

#include "support/utils.h"
#include "support/scan_serial_impl.h"

using namespace TestUtils;

struct test_inclusive
{
    template <typename Policy, typename Iterator1, typename Iterator2, typename Iterator3, typename Size, typename T>
    std::enable_if_t<!TestUtils::is_reverse_v<Iterator1>>
    operator()(Policy&& exec, Iterator1 in_first, Iterator1 in_last, Iterator2 out_first, Iterator2 out_last,
               Iterator3 expected_first, Iterator3, Size n, T init, T trash)
    {
        inclusive_scan_serial(in_first, in_last, expected_first);
        auto orr = std::inclusive_scan(std::forward<Policy>(exec), in_first, in_last, out_first);
        EXPECT_TRUE(out_last == orr, "inclusive_scan returned wrong iterator");
        EXPECT_EQ_N(expected_first, out_first, n, "wrong result from inclusive_scan");
        std::fill_n(out_first, n, trash);
    }
    template <typename Policy, typename Iterator1, typename Iterator2, typename Iterator3, typename Size, typename T>
    std::enable_if_t<TestUtils::is_reverse_v<Iterator1>>
    operator()(Policy&&, Iterator1, Iterator1, Iterator2, Iterator2, Iterator3, Iterator3, Size, T, T) {}
};

struct test_exclusive
{
    template <typename Policy, typename Iterator1, typename Iterator2, typename Iterator3, typename Size, typename T>
    std::enable_if_t<!TestUtils::is_reverse_v<Iterator1>>
    operator()(Policy&& exec, Iterator1 in_first, Iterator1 in_last, Iterator2 out_first, Iterator2 out_last,
               Iterator3 expected_first, Iterator3, Size n, T init, T trash)
    {
        exclusive_scan_serial(in_first, in_last, expected_first, init);
        auto orr = std::exclusive_scan(std::forward<Policy>(exec), in_first, in_last, out_first, init);
        EXPECT_TRUE(out_last == orr, "exclusive_scan returned wrong iterator");
        EXPECT_EQ_N(expected_first, out_first, n, "wrong result from exclusive_scan");
        std::fill_n(out_first, n, trash);
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

    // Only n=20000 — above the CPU single-group cutoff of 16384
    constexpr size_t n = 20000;
    Sequence<T> in(n, convert);
    Sequence<T> expected(n);
    Sequence<T> out(n, [&](std::int32_t) { return trash; });

    invoke_on_all_policies<0>()(test_inclusive(), in.begin(), in.end(), out.begin(),
                                out.end(), expected.begin(), expected.end(), in.size(), init, trash);
    invoke_on_all_policies<1>()(test_exclusive(), in.begin(), in.end(), out.begin(),
                                out.end(), expected.begin(), expected.end(), in.size(), init, trash);

    return done();
}
