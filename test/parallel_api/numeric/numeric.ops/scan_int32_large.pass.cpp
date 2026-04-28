// -*- C++ -*-
//===-- scan_int32_large.pass.cpp -----------------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Isolation test: int32_t scan with large sizes only (70M+). Tests size dependency.

#include "support/test_config.h"

#include _PSTL_TEST_HEADER(execution)
#include _PSTL_TEST_HEADER(numeric)

#include "support/utils.h"
#include "support/scan_serial_impl.h"

using namespace TestUtils;

struct test_inclusive_scan_int32
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
    operator()(Policy&&, Iterator1, Iterator1, Iterator2, Iterator2, Iterator3, Iterator3, Size, T, T)
    {
    }
};

struct test_exclusive_scan_int32
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
    operator()(Policy&&, Iterator1, Iterator1, Iterator2, Iterator2, Iterator3, Iterator3, Size, T, T)
    {
    }
};

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT && !ONEDPL_FPGA_DEVICE
    using T = std::int32_t;
    T init = 0;
    T trash = -666;
    auto convert = [](std::uint32_t k) { return T((k % 991 + 1) ^ (k % 997 + 2)); };

    // Large sizes only
    for (unsigned long n = 70000000; n <= 100000000; n += 10000000)
    {
        Sequence<T> in(n, convert);
        Sequence<T> expected(n);
        Sequence<T> out(n, [&](std::int32_t) { return trash; });

        invoke_on_all_hetero_policies<0>()(test_inclusive_scan_int32(), in.begin(), in.end(), out.begin(),
                                           out.end(), expected.begin(), expected.end(), in.size(), init, trash);
        invoke_on_all_hetero_policies<1>()(test_exclusive_scan_int32(), in.begin(), in.end(), out.begin(),
                                           out.end(), expected.begin(), expected.end(), in.size(), init, trash);
    }
#else
    // Nothing to test without SYCL backend at large sizes
#endif

    return done();
}
