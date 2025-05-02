// -*- C++ -*-
//===-- transform_scan.pass.cpp -------------------------------------------===//
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
#include _PSTL_TEST_HEADER(algorithm)

#include "support/utils.h"

#if  !defined(_PSTL_TEST_TRANSFORM_INCLUSIVE_SCAN) && !defined(_PSTL_TEST_TRANSFORM_EXCLUSIVE_SCAN)
#define _PSTL_TEST_TRANSFORM_INCLUSIVE_SCAN
#define _PSTL_TEST_TRANSFORM_EXCLUSIVE_SCAN
#endif

using namespace TestUtils;

// Most of the framework required for testing inclusive and exclusive transform-scans is identical,
// so the tests for both are in this file.  Which is being tested is controlled by the global
// flag inclusive, which is set to each alternative by main().

template <typename Iterator, typename Size, typename T>
void
check_and_reset(Iterator expected_first, Iterator out_first, Size n, T trash)
{
    EXPECT_EQ_N(expected_first, out_first, n, "wrong result from transform_..._scan");
    ::std::fill_n(out_first, n, trash);
}

template <typename Type>
struct test_transform_exclusive_scan
{
    template <typename Policy, typename InputIterator, typename OutputIterator, typename Size, typename UnaryOp,
              typename T, typename BinaryOp>
    ::std::enable_if_t<!TestUtils::is_reverse_v<InputIterator>>
    operator()(Policy&& exec, InputIterator first, InputIterator last, OutputIterator out_first,
               OutputIterator out_last, OutputIterator expected_first, OutputIterator /* expected_last */, Size n,
               UnaryOp unary_op, T init, BinaryOp binary_op, T trash)
    {
        using namespace std;

        transform_exclusive_scan(oneapi::dpl::execution::seq, first, last, expected_first, init, binary_op, unary_op);
        auto orr2 = transform_exclusive_scan(exec, first, last, out_first, init, binary_op, unary_op);
        EXPECT_TRUE(out_last == orr2, "transform_exclusive_scan returned wrong iterator");
        EXPECT_EQ_N(expected_first, out_first, n, "wrong result from transform_exclusive_scan");
        ::std::fill_n(out_first, n, trash);
    }

    template <typename Policy, typename InputIterator, typename OutputIterator, typename Size, typename UnaryOp,
              typename T, typename BinaryOp>
    ::std::enable_if_t<TestUtils::is_reverse_v<InputIterator>>
    operator()(Policy&& /* exec */, InputIterator /* first */, InputIterator /* last */, OutputIterator /* out_first */,
               OutputIterator /* out_last */, OutputIterator /* expected_first */, OutputIterator /* expected_last */, Size /* n */,
               UnaryOp /* unary_op */, T /* init */, BinaryOp /* binary_op */, T /* trash */)
    {
    }
};

template <typename Type>
struct test_transform_inclusive_scan_init
{
    template <typename Policy, typename InputIterator, typename OutputIterator, typename Size, typename UnaryOp,
              typename T, typename BinaryOp>
    ::std::enable_if_t<!TestUtils::is_reverse_v<InputIterator>>
    operator()(Policy&& exec, InputIterator first, InputIterator last, OutputIterator out_first,
               OutputIterator out_last, OutputIterator expected_first, OutputIterator /* expected_last */, Size n,
               UnaryOp unary_op, T init, BinaryOp binary_op, T trash)
    {
        using namespace std;

        transform_inclusive_scan(oneapi::dpl::execution::seq, first, last, expected_first, binary_op, unary_op, init);
        auto orr2 = transform_inclusive_scan(exec, first, last, out_first, binary_op, unary_op, init);
        EXPECT_TRUE(out_last == orr2, "transform_inclusive_scan returned wrong iterator");
        EXPECT_EQ_N(expected_first, out_first, n, "wrong result from transform_inclusive_scan");
        ::std::fill_n(out_first, n, trash);
    }

    template <typename Policy, typename InputIterator, typename OutputIterator, typename Size, typename UnaryOp,
              typename T, typename BinaryOp>
    ::std::enable_if_t<TestUtils::is_reverse_v<InputIterator>>
    operator()(Policy&& /* exec */, InputIterator /* first */, InputIterator /* last */, OutputIterator /* out_first */,
               OutputIterator /* out_last */, OutputIterator /* expected_first */, OutputIterator /* expected_last */, Size /* n */,
               UnaryOp /* unary_op */, T /* init */, BinaryOp /* binary_op */, T /* trash */)
    {
    }
};

template <typename Type>
struct test_transform_inclusive_scan
{
    template <typename Policy, typename InputIterator, typename OutputIterator, typename Size, typename UnaryOp,
              typename T, typename BinaryOp>
    ::std::enable_if_t<!TestUtils::is_reverse_v<InputIterator>>
    operator()(Policy&& exec, InputIterator first, InputIterator last, OutputIterator out_first,
               OutputIterator out_last, OutputIterator expected_first, OutputIterator /* expected_last */, Size n,
               UnaryOp unary_op, T /* init */, BinaryOp binary_op, T trash)
    {
        using namespace std;

        if (n > 0)
        {
            transform_inclusive_scan(oneapi::dpl::execution::seq, first, last, expected_first, binary_op, unary_op);
            auto orr2 = transform_inclusive_scan(exec, first, last, out_first, binary_op, unary_op);
            EXPECT_TRUE(out_last == orr2, "transform_inclusive_scan returned wrong iterator");
            EXPECT_EQ_N(expected_first, out_first, n, "wrong result from transform_inclusive_scan");
            ::std::fill_n(out_first, n, trash);
        }
    }

    template <typename Policy, typename InputIterator, typename OutputIterator, typename Size, typename UnaryOp,
              typename T, typename BinaryOp>
    ::std::enable_if_t<TestUtils::is_reverse_v<InputIterator>>
    operator()(Policy&& /* exec */, InputIterator /* first */, InputIterator /* last */, OutputIterator /* out_first */,
               OutputIterator /* out_last */, OutputIterator /* expected_first */, OutputIterator /* expected_last */, Size /* n */,
               UnaryOp /* unary_op */, T /* init */, BinaryOp /* binary_op */, T /* trash */)
    {
    }
};

const std::uint32_t encryption_mask = 0x314;

template <typename InputIterator, typename OutputIterator, typename UnaryOperation, typename T,
          typename BinaryOperation>
::std::pair<OutputIterator, T>
transform_inclusive_scan_serial(InputIterator first, InputIterator last, OutputIterator result, UnaryOperation unary_op,
                                T init, BinaryOperation binary_op) noexcept
{
    for (; first != last; ++first, ++result)
    {
        init = binary_op(init, unary_op(*first));
        *result = init;
    }
    return ::std::make_pair(result, init);
}

template <typename InputIterator, typename OutputIterator, typename UnaryOperation, typename T,
          typename BinaryOperation>
::std::pair<OutputIterator, T>
transform_exclusive_scan_serial(InputIterator first, InputIterator last, OutputIterator result, UnaryOperation unary_op,
                                T init, BinaryOperation binary_op) noexcept
{
    for (; first != last; ++first, ++result)
    {
        *result = init;
        init = binary_op(init, unary_op(*first));
    }
    return ::std::make_pair(result, init);
}

template <typename In, typename Out, typename UnaryOp, typename BinaryOp>
void
test(UnaryOp unary_op, Out init, BinaryOp binary_op, Out trash)
{
    for (size_t n = 1; n <= 100000; n = n <= 16 ? n + 1 : size_t(3.1415 * n))
    {
        Sequence<In> in(n, [](size_t k) { return In(k ^ encryption_mask); });

        Out tmp = init;
        Sequence<Out> expected1(n, [&](size_t k) -> Out {
            Out val = tmp;
            tmp = binary_op(tmp, unary_op(in[k]));
            return val;
        });

        tmp = init;
        Sequence<Out> expected2(n, [&](size_t k) -> Out {
            tmp = binary_op(tmp, unary_op(in[k]));
            return tmp;
        });

        Sequence<Out> out(n, [&](size_t) { return trash; });

#ifdef _PSTL_TEST_TRANSFORM_INCLUSIVE_SCAN
        transform_inclusive_scan_serial(in.cbegin(), in.cend(), out.fbegin(), unary_op, init, binary_op);
        check_and_reset(expected2.begin(), out.begin(), out.size(), trash);
        invoke_on_all_policies<1>()(test_transform_inclusive_scan_init<In>(), in.begin(), in.end(), out.begin(),
                                    out.end(), expected2.begin(), expected2.end(), in.size(), unary_op, init,
                                    binary_op, trash);
        invoke_on_all_policies<2>()(test_transform_inclusive_scan<In>(), in.begin(), in.end(), out.begin(), out.end(),
                                    expected2.begin(), expected2.end(), in.size(), unary_op, init, binary_op, trash);
#if !ONEDPL_FPGA_DEVICE
        invoke_on_all_policies<3>()(test_transform_inclusive_scan_init<In>(), in.cbegin(), in.cend(), out.begin(),
                                    out.end(), expected2.begin(), expected2.end(), in.size(), unary_op, init,
                                    binary_op, trash);
        invoke_on_all_policies<4>()(test_transform_inclusive_scan<In>(), in.cbegin(), in.cend(), out.begin(),
                                    out.end(), expected2.begin(), expected2.end(), in.size(), unary_op, init,
                                    binary_op, trash);
#endif
#endif // _PSTL_TEST_TRANSFORM_INCLUSIVE_SCAN
#ifdef _PSTL_TEST_TRANSFORM_EXCLUSIVE_SCAN
        transform_exclusive_scan_serial(in.cbegin(), in.cend(), out.fbegin(), unary_op, init, binary_op);
        check_and_reset(expected1.begin(), out.begin(), out.size(), trash);
        invoke_on_all_policies<5>()(test_transform_exclusive_scan<In>(), in.begin(), in.end(), out.begin(), out.end(),
                                    expected1.begin(), expected1.end(), in.size(), unary_op, init, binary_op, trash);
#if !ONEDPL_FPGA_DEVICE
        invoke_on_all_policies<6>()(test_transform_exclusive_scan<In>(), in.cbegin(), in.cend(), out.begin(),
                                    out.end(), expected1.begin(), expected1.end(), in.size(), unary_op, init,
                                    binary_op, trash);
#endif
        ::std::copy(in.begin(), in.end(), out.begin());
        invoke_on_all_policies<13>()(test_transform_exclusive_scan<In>(), out.begin(), out.end(), out.begin(), out.end(),
                                    expected1.begin(), expected1.end(), in.size(), unary_op, init, binary_op, trash);
#endif // _PSTL_TEST_TRANSFORM_EXCLUSIVE_SCAN
    }
}

template <typename In, typename Out, typename UnaryOp, typename BinaryOp>
void
test_matrix(UnaryOp unary_op, Out init, BinaryOp binary_op, Out trash)
{
    for (size_t n = 0; n <= 100000; n = n <= 16 ? n + 1 : size_t(3.1415 * n))
    {
        Sequence<In> in(n, [](size_t k) { return In(k, k + 1); });

        Sequence<Out> out(n, [&](size_t) { return trash; });
        Sequence<Out> expected(n, [&](size_t) { return trash; });

#ifdef _PSTL_TEST_TRANSFORM_INCLUSIVE_SCAN
        invoke_on_all_policies<7>()(test_transform_inclusive_scan_init<In>(), in.begin(), in.end(), out.begin(),
                                    out.end(), expected.begin(), expected.end(), in.size(), unary_op, init, binary_op,
                                    trash);
        invoke_on_all_policies<8>()(test_transform_inclusive_scan_init<In>(), in.cbegin(), in.cend(), out.begin(),
                                    out.end(), expected.begin(), expected.end(), in.size(), unary_op, init, binary_op,
                                    trash);
        invoke_on_all_policies<9>()(test_transform_inclusive_scan<In>(), in.begin(), in.end(), out.begin(), out.end(),
                                     expected.begin(), expected.end(), in.size(), unary_op, init, binary_op, trash);
        invoke_on_all_policies<10>()(test_transform_inclusive_scan<In>(), in.cbegin(), in.cend(), out.begin(),
                                     out.end(), expected.begin(), expected.end(), in.size(), unary_op, init, binary_op,
                                     trash);
#endif
#ifdef _PSTL_TEST_TRANSFORM_EXCLUSIVE_SCAN
#if !TEST_GCC10_EXCLUSIVE_SCAN_BROKEN
        invoke_on_all_policies<11>()(test_transform_exclusive_scan<In>(), in.begin(), in.end(), out.begin(), out.end(),
                                    expected.begin(), expected.end(), in.size(), unary_op, init, binary_op, trash);
        invoke_on_all_policies<12>()(test_transform_exclusive_scan<In>(), in.cbegin(), in.cend(), out.begin(),
                                    out.end(), expected.begin(), expected.end(), in.size(), unary_op, init, binary_op,
                                    trash);
#endif
#endif
    }
}

int
main()
{
#if !_PSTL_ICC_19_TEST_SIMD_UDS_WINDOWS_RELEASE_BROKEN
    test_matrix<Matrix2x2<std::int32_t>, Matrix2x2<std::int32_t>>([](const Matrix2x2<std::int32_t> x) { return x; },
                                                        Matrix2x2<std::int32_t>(), multiply_matrix<std::int32_t>(),
                                                        Matrix2x2<std::int32_t>(-666, 666));
#endif
    test<std::int32_t, std::uint32_t>([](std::int32_t x) { return x++; }, -123, [](std::int32_t x, std::int32_t y) { return x + y; }, 666);

    return done();
}
