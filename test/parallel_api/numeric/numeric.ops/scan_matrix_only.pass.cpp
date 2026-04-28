// -*- C++ -*-
//===-- scan_matrix_only.pass.cpp -----------------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Isolation test: Only Matrix2x2<int32_t> scan to confirm type-specific crash.

#include "support/test_config.h"

#include _PSTL_TEST_HEADER(execution)
#include _PSTL_TEST_HEADER(numeric)

#include "support/utils.h"
#include "support/scan_serial_impl.h"

using namespace TestUtils;

template <typename Type>
struct test_inclusive_scan_matrix
{
    template <typename Policy, typename Iterator1, typename Iterator2, typename Iterator3, typename Size, typename T,
              typename BinaryOp>
    std::enable_if_t<!TestUtils::is_reverse_v<Iterator1>>
    operator()(Policy&& exec, Iterator1 in_first, Iterator1 in_last, Iterator2 out_first, Iterator2 out_last,
               Iterator3 expected_first, Iterator3, Size n, T init, BinaryOp binary_op, T trash)
    {
        inclusive_scan_serial(in_first, in_last, expected_first, binary_op, init);
        auto orr = std::inclusive_scan(std::forward<Policy>(exec), in_first, in_last, out_first, binary_op, init);
        EXPECT_TRUE(out_last == orr, "inclusive_scan returned wrong iterator");
        EXPECT_EQ_N(expected_first, out_first, n, "wrong result from inclusive_scan");
        std::fill_n(out_first, n, trash);
    }

    template <typename Policy, typename Iterator1, typename Iterator2, typename Iterator3, typename Size, typename T,
              typename BinaryOp>
    std::enable_if_t<TestUtils::is_reverse_v<Iterator1>>
    operator()(Policy&&, Iterator1, Iterator1, Iterator2, Iterator2, Iterator3, Iterator3, Size, T, BinaryOp, T)
    {
    }
};

template <typename Type>
struct test_exclusive_scan_matrix
{
    template <typename Policy, typename Iterator1, typename Iterator2, typename Iterator3, typename Size, typename T,
              typename BinaryOp>
    std::enable_if_t<!TestUtils::is_reverse_v<Iterator1>>
    operator()(Policy&& exec, Iterator1 in_first, Iterator1 in_last, Iterator2 out_first, Iterator2 out_last,
               Iterator3 expected_first, Iterator3, Size n, T init, BinaryOp binary_op, T trash)
    {
        exclusive_scan_serial(in_first, in_last, expected_first, init, binary_op);
        auto orr = std::exclusive_scan(std::forward<Policy>(exec), in_first, in_last, out_first, init, binary_op);
        EXPECT_TRUE(out_last == orr, "exclusive_scan returned wrong iterator");
        EXPECT_EQ_N(expected_first, out_first, n, "wrong result from exclusive_scan");
        std::fill_n(out_first, n, trash);
    }

    template <typename Policy, typename Iterator1, typename Iterator2, typename Iterator3, typename Size, typename T,
              typename BinaryOp>
    std::enable_if_t<TestUtils::is_reverse_v<Iterator1>>
    operator()(Policy&&, Iterator1, Iterator1, Iterator2, Iterator2, Iterator3, Iterator3, Size, T, BinaryOp, T)
    {
    }
};

int
main()
{
    using T = Matrix2x2<std::int32_t>;
    T init;
    T trash(-666, 666);
    multiply_matrix<std::int32_t> binary_op;

    for (size_t n = 0; n <= 100000; n = n <= 16 ? n + 1 : size_t(3.1415 * n))
    {
        Sequence<T> in(n, [](size_t k) { return T(k, k + 1); });
        Sequence<T> out(n, [&](size_t) { return trash; });
        Sequence<T> expected(n, [&](size_t) { return trash; });

        invoke_on_all_policies<0>()(test_inclusive_scan_matrix<T>(), in.begin(), in.end(), out.begin(),
                                    out.end(), expected.begin(), expected.end(), in.size(), init, binary_op, trash);
        invoke_on_all_policies<1>()(test_inclusive_scan_matrix<T>(), in.cbegin(), in.cend(), out.begin(),
                                    out.end(), expected.begin(), expected.end(), in.size(), init, binary_op, trash);
#if !TEST_GCC10_EXCLUSIVE_SCAN_BROKEN
        invoke_on_all_policies<2>()(test_exclusive_scan_matrix<T>(), in.begin(), in.end(), out.begin(),
                                    out.end(), expected.begin(), expected.end(), in.size(), init, binary_op, trash);
        invoke_on_all_policies<3>()(test_exclusive_scan_matrix<T>(), in.cbegin(), in.cend(), out.begin(),
                                    out.end(), expected.begin(), expected.end(), in.size(), init, binary_op, trash);
#endif
    }

    return done();
}
