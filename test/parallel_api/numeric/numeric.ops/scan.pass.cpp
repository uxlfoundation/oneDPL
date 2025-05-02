// -*- C++ -*-
//===-- scan.pass.cpp -----------------------------------------------------===//
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

#include _PSTL_TEST_HEADER(execution)
#include _PSTL_TEST_HEADER(numeric)

#include "support/utils.h"
#include "support/scan_serial_impl.h"

#include <random>
#include <algorithm>
#include <cstdint>

#if  !defined(_PSTL_TEST_INCLUSIVE_SCAN) && !defined(_PSTL_TEST_EXCLUSIVE_SCAN)
#define _PSTL_TEST_INCLUSIVE_SCAN
#define _PSTL_TEST_EXCLUSIVE_SCAN
#endif

using namespace TestUtils;

// Most of the framework required for testing inclusive and exclusive scan is identical,
// so the tests for both are in this file.  Which is being tested is controlled by the global
// flag inclusive, which is set to each alternative by main().
//static bool inclusive;

template <typename Type>
struct test_inclusive_scan_with_plus
{
    template <typename Policy, typename Iterator1, typename Iterator2, typename Iterator3, typename Size, typename T>
    void
    operator()(Policy&& exec, Iterator1 in_first, Iterator1 in_last, Iterator2 out_first, Iterator2 out_last,
               Iterator3 expected_first, Iterator3 /* expected_last */, Size n, T /* init */, T trash)
    {
        using namespace std;

        inclusive_scan_serial(in_first, in_last, expected_first);
        auto orr = inclusive_scan(exec, in_first, in_last, out_first);
        EXPECT_TRUE(out_last == orr, "inclusive_scan returned wrong iterator");
        EXPECT_EQ_N(expected_first, out_first, n, "wrong result from inclusive_scan");
        ::std::fill_n(out_first, n, trash);
    }
};

template <typename Type>
struct test_exclusive_scan_with_plus
{
    template <typename Policy, typename Iterator1, typename Iterator2, typename Iterator3, typename Size, typename T>
    void
    operator()(Policy&& exec, Iterator1 in_first, Iterator1 in_last, Iterator2 out_first, Iterator2 out_last,
               Iterator3 expected_first, Iterator3 /* expected_last */, Size n, T init, T trash)
    {
        using namespace std;

        exclusive_scan_serial(in_first, in_last, expected_first, init);
        auto orr = exclusive_scan(exec, in_first, in_last, out_first, init);
        EXPECT_TRUE(out_last == orr, "exclusive_scan returned wrong iterator");
        EXPECT_EQ_N(expected_first, out_first, n, "wrong result from exclusive_scan");
        ::std::fill_n(out_first, n, trash);
    }
};

template <typename T, typename Convert>
void
test_with_plus(T init, T trash, Convert convert)
{
    for (size_t n = 0; n <= 100000; n = n <= 16 ? n + 1 : size_t(3.1415 * n))
    {
        Sequence<T> in(n, convert);
        Sequence<T> expected(in);
        Sequence<T> out(n, [&](std::int32_t) { return trash; });

#ifdef _PSTL_TEST_INCLUSIVE_SCAN

        invoke_on_all_policies<0>()(test_inclusive_scan_with_plus<T>(), in.begin(), in.end(), out.begin(), out.end(),
                                    expected.begin(), expected.end(), in.size(), init, trash);
        invoke_on_all_policies<1>()(test_inclusive_scan_with_plus<T>(), in.cbegin(), in.cend(), out.begin(),
                                    out.end(), expected.begin(), expected.end(), in.size(), init, trash);
#endif

#ifdef _PSTL_TEST_EXCLUSIVE_SCAN

        invoke_on_all_policies<2>()(test_exclusive_scan_with_plus<T>(), in.begin(), in.end(), out.begin(),
                                    out.end(), expected.begin(), expected.end(), in.size(), init, trash);
        invoke_on_all_policies<3>()(test_exclusive_scan_with_plus<T>(), in.cbegin(), in.cend(), out.begin(), out.end(),
                                    expected.begin(), expected.end(), in.size(), init, trash);
#endif
    }

#if TEST_DPCPP_BACKEND_PRESENT && !ONEDPL_FPGA_DEVICE
    // testing of large number of items may take too much time in debug mode
    unsigned long n =
#if PSTL_USE_DEBUG
        70000000;
#else
        100000000;
#endif

    Sequence<T> in(n, convert);
    Sequence<T> expected(in);
    Sequence<T> out(n, [&](std::int32_t) { return trash; });
#ifdef _PSTL_TEST_INCLUSIVE_SCAN
    invoke_on_all_hetero_policies<4>()(test_inclusive_scan_with_plus<T>(), in.begin(), in.end(), out.begin(), out.end(),
                                expected.begin(), expected.end(), in.size(), init, trash);
#endif
#ifdef _PSTL_TEST_EXCLUSIVE_SCAN
    invoke_on_all_hetero_policies<5>()(test_exclusive_scan_with_plus<T>(), in.begin(), in.end(), out.begin(),
                                out.end(), expected.begin(), expected.end(), in.size(), init, trash);
#endif
#endif // TEST_DPCPP_BACKEND_PRESENT && !ONEDPL_FPGA_DEVICE
}

template <typename Type>
struct test_inclusive_scan_with_binary_op
{
    template <typename Policy, typename Iterator1, typename Iterator2, typename Iterator3, typename Size, typename T,
              typename BinaryOp>
    ::std::enable_if_t<!TestUtils::is_reverse_v<Iterator1>>
    operator()(Policy&& exec, Iterator1 in_first, Iterator1 in_last, Iterator2 out_first, Iterator2 out_last,
               Iterator3 expected_first, Iterator3 /* expected_last */, Size n, T init, BinaryOp binary_op, T trash)
    {
        using namespace std;

        inclusive_scan_serial(in_first, in_last, expected_first, binary_op, init);
        auto orr = inclusive_scan(exec, in_first, in_last, out_first, binary_op, init);

        EXPECT_TRUE(out_last == orr, "inclusive_scan with binary operator returned wrong iterator");
        EXPECT_EQ_N(expected_first, out_first, n, "wrong result from inclusive_scan with binary operator");
        ::std::fill_n(out_first, n, trash);
    }

    template <typename Policy, typename Iterator1, typename Iterator2, typename Iterator3, typename Size, typename T,
              typename BinaryOp>
    ::std::enable_if_t<!TestUtils::is_reverse_v<Iterator1>>
    operator()(Policy&& exec, Iterator1 in_first, Iterator1 in_last, Iterator2 out_first, Iterator2 out_last,
               Iterator3 expected_first, Iterator3 /* expected_last */, Size n, BinaryOp binary_op, T trash)
    {
        using namespace std;

        inclusive_scan_serial(in_first, in_last, expected_first, binary_op);
        auto orr = inclusive_scan(exec, in_first, in_last, out_first, binary_op);

        EXPECT_TRUE(out_last == orr, "inclusive_scan with binary operator without init returned wrong iterator");
        EXPECT_EQ_N(expected_first, out_first, n, "wrong result from inclusive_scan with binary operator without init");
        ::std::fill_n(out_first, n, trash);
    }

    template <typename Policy, typename Iterator1, typename Iterator2, typename Iterator3, typename Size, typename T,
              typename BinaryOp>
    ::std::enable_if_t<TestUtils::is_reverse_v<Iterator1>>
    operator()(Policy&& /* exec */, Iterator1 /* in_first */, Iterator1 /* in_last */, Iterator2 /* out_first */, Iterator2 /* out_last */,
               Iterator3 /* expected_first */, Iterator3 /* expected_last */, Size /* n */, BinaryOp /* binary_op */, T /* trash */)
    {
    }

    template <typename Policy, typename Iterator1, typename Iterator2, typename Iterator3, typename Size, typename T,
              typename BinaryOp>
    ::std::enable_if_t<TestUtils::is_reverse_v<Iterator1>>
    operator()(Policy&& /* exec */, Iterator1 /* in_first */, Iterator1 /* in_last */, Iterator2 /* out_first */, Iterator2 /* out_last */,
               Iterator3 /* expected_first */, Iterator3 /* expected_last */, Size /* n */, T /* init */, BinaryOp /* binary_op */, T /* trash */)
    {
    }
};

template <typename Type>
struct test_exclusive_scan_with_binary_op
{
    template <typename Policy, typename Iterator1, typename Iterator2, typename Iterator3, typename Size, typename T,
              typename BinaryOp>
    ::std::enable_if_t<!TestUtils::is_reverse_v<Iterator1>>
    operator()(Policy&& exec, Iterator1 in_first, Iterator1 in_last, Iterator2 out_first, Iterator2 out_last,
               Iterator3 expected_first, Iterator3 /* expected_last */, Size n, T init, BinaryOp binary_op, T trash)
    {
        using namespace std;

        exclusive_scan_serial(in_first, in_last, expected_first, init, binary_op);

        auto orr = exclusive_scan(exec, in_first, in_last, out_first, init, binary_op);

        EXPECT_TRUE(out_last == orr, "exclusive_scan with binary operator returned wrong iterator");
        EXPECT_EQ_N(expected_first, out_first, n, "wrong result from exclusive_scan with binary operator");
        ::std::fill_n(out_first, n, trash);
    }

    template <typename Policy, typename Iterator1, typename Iterator2, typename Iterator3, typename Size, typename T,
              typename BinaryOp>
    ::std::enable_if_t<TestUtils::is_reverse_v<Iterator1>>
    operator()(Policy&& /* exec */, Iterator1 /* in_first */, Iterator1 /* in_last */, Iterator2 /* out_first */, Iterator2 /* out_last */,
               Iterator3 /* expected_first */, Iterator3 /* expected_last */, Size /* n */, T /* init */, BinaryOp /* binary_op */, T /* trash */)
    {
    }
};

template <typename In, typename Out, typename BinaryOp>
void
test_matrix(Out init, BinaryOp binary_op, Out trash)
{
    for (size_t n = 0; n <= 100000; n = n <= 16 ? n + 1 : size_t(3.1415 * n))
    {
        Sequence<In> in(n, [](size_t k) { return In(k, k + 1); });

        Sequence<Out> out(n, [&](size_t) { return trash; });
        Sequence<Out> expected(n, [&](size_t) { return trash; });

        auto __scan_invoker = [&](Sequence<Out>& out) {
#ifdef _PSTL_TEST_INCLUSIVE_SCAN
        invoke_on_all_policies<4>()(test_inclusive_scan_with_binary_op<In>(), in.begin(), in.end(), out.begin(),
                                    out.end(), expected.begin(), expected.end(), in.size(), init, binary_op, trash);
        invoke_on_all_policies<5>()(test_inclusive_scan_with_binary_op<In>(), in.cbegin(), in.cend(), out.begin(),
                                    out.end(), expected.begin(), expected.end(), in.size(), init, binary_op, trash);
        invoke_on_all_policies<6>()(test_inclusive_scan_with_binary_op<In>(), in.begin(), in.end(), out.begin(),
                                    out.end(), expected.begin(), expected.end(), in.size(), binary_op, trash);
        invoke_on_all_policies<7>()(test_inclusive_scan_with_binary_op<In>(), in.cbegin(), in.cend(), out.begin(),
                                    out.end(), expected.begin(), expected.end(), in.size(), binary_op, trash);
#endif

#ifdef _PSTL_TEST_EXCLUSIVE_SCAN
#if !TEST_GCC10_EXCLUSIVE_SCAN_BROKEN
        invoke_on_all_policies<8>()(test_exclusive_scan_with_binary_op<In>(), in.begin(), in.end(), out.begin(),
                                    out.end(), expected.begin(), expected.end(), in.size(), init, binary_op, trash);
        invoke_on_all_policies<9>()(test_exclusive_scan_with_binary_op<In>(), in.cbegin(), in.cend(), out.begin(),
                                    out.end(), expected.begin(), expected.end(), in.size(), init, binary_op, trash);
#endif
#endif
        };

        //perform regular a scan algorithm
        __scan_invoker(out);

        //perform an in-place scan algorithm
        __scan_invoker(in);
    }
}

template <typename T>
void
test_with_multiplies()
{
#if TEST_DPCPP_BACKEND_PRESENT
    T trash = 666;
    T init = 1;
    const std::size_t custom_item_count = 10;

    for (size_t n = custom_item_count; n <= 100000; n = n <= 16 ? n + 1 : size_t(3.1415 * n))
    {
        Sequence<T> out(n, [&](size_t) { return trash; });
        Sequence<T> expected(n, [&](size_t) { return trash; });

        Sequence<T> in(n, [](size_t k) { return 1; });
        std::size_t counter = 0;
        std::generate_n(in.begin(), custom_item_count, [&counter]() { return (counter++) % 3 + 2; } );
        std::default_random_engine gen{42};
        std::shuffle(in.begin(), in.end(), gen);

#ifdef _PSTL_TEST_INCLUSIVE_SCAN
        invoke_on_all_hetero_policies<20>()(test_inclusive_scan_with_binary_op<T>(), in.begin(), in.end(),
                                            out.begin(), out.end(), expected.begin(), expected.end(), in.size(),
                                            init, std::multiplies{}, trash);
#endif
#ifdef _PSTL_TEST_EXCLUSIVE_SCAN
        invoke_on_all_hetero_policies<21>()(test_exclusive_scan_with_binary_op<T>(), in.begin(), in.end(), out.begin(),
                                            out.end(), expected.begin(), expected.end(), in.size(),
                                            init, std::multiplies{}, trash);
#endif
    }
#endif // TEST_DPCPP_BACKEND_PRESENT
}

int
main()
{
#if !_PSTL_ICC_19_TEST_SIMD_UDS_WINDOWS_RELEASE_BROKEN
    // Test with highly restricted type and associative but not commutative operation
    test_matrix<Matrix2x2<std::int32_t>, Matrix2x2<std::int32_t>>(Matrix2x2<std::int32_t>(), multiply_matrix<std::int32_t>(),
                                                            Matrix2x2<std::int32_t>(-666, 666));
#endif

    // Since the implicit "+" forms of the scan delegate to the generic forms,
    // there's little point in using a highly restricted type, so just use double.
    test_with_plus<float64_t>(0.0, -666.0, [](std::uint32_t k) { return float64_t((k % 991 + 1) ^ (k % 997 + 2)); });
    test_with_plus<std::int32_t>(0.0, -666.0, [](std::uint32_t k) { return std::int32_t((k % 991 + 1) ^ (k % 997 + 2)); });

    test_with_multiplies<std::uint64_t>();

    return done();
}
