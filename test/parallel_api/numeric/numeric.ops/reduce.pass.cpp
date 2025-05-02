// -*- C++ -*-
//===-- reduce.pass.cpp ---------------------------------------------------===//
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

using namespace TestUtils;

template <typename Type>
struct test_long_reduce
{
    template <typename Policy, typename Iterator, typename T, typename BinaryOp>
    void
    operator()(Policy&& exec, Iterator first, Iterator last, T init, BinaryOp binary, T expected)
    {
        T result_r = ::std::reduce(exec, first, last, init, binary);
        EXPECT_EQ(expected, result_r, "bad result from reduce(exec, first, last, init, binary_op)");
    }
};

template <typename T, typename BinaryOp, typename F>
void
test_long_form(T init, BinaryOp binary_op, F f)
{
    // Try sequences of various lengths
    for (size_t n = 0; n <= 100000; n = n <= 16 ? n + 1 : size_t(3.1415 * n))
    {
        T expected(init);
        Sequence<T> in(n, [n, f](size_t k) { return f((std::int32_t(k ^ n) % 1000 - 500)); });
        for (size_t k = 0; k < n; ++k)
            expected = binary_op(expected, in[k]);

        using namespace std;

        T result = transform_reduce_serial(in.cfbegin(), in.cfend(), init, binary_op, [](const T& t) { return t; });
        EXPECT_EQ(expected, result, "bad result from reduce(first, last, init, binary_op_op)");

        invoke_on_all_policies<0>()(test_long_reduce<T>(), in.begin(), in.end(), init, binary_op, expected);
        invoke_on_all_policies<1>()(test_long_reduce<T>(), in.cbegin(), in.cend(), init, binary_op, expected);
    }
}

struct test_short_reduce
{
    template <typename Policy, typename Iterator>
    void
    operator()(Policy&& exec, Iterator first, Iterator last, Sum init, Sum expected)
    {
        using namespace std;

        Sum r0 = init + reduce(exec, first, last);
        EXPECT_EQ(expected, r0, "bad result from reduce(exec, first, last)");
    }
};

struct test_short_reduce_init
{
    template <typename Policy, typename Iterator>
    void
    operator()(Policy&& exec, Iterator first, Iterator last, Sum init, Sum expected)
    {
        using namespace std;

        Sum r1 = reduce(exec, first, last, init);
        EXPECT_EQ(expected, r1, "bad result from reduce(exec, first, last, init)");
    }
};

// Test forms of reduce(...) that omit the binary_op or init operands.
void
test_short_forms()
{
    for (size_t n = 0; n <= 100000; n = n <= 16 ? n + 1 : size_t(3.1415 * n))
    {
        Sum init(42, OddTag());
        Sum expected(init);
        Sequence<Sum> in(n, [n](size_t k) { return Sum((std::int32_t(k ^ n) % 1000 - 500), OddTag()); });
        for (size_t k = 0; k < n; ++k)
            expected = expected + in[k];
        invoke_on_all_policies<2>()(test_short_reduce(), in.begin(), in.end(), init, expected);
        invoke_on_all_policies<3>()(test_short_reduce_init(), in.begin(), in.end(), init, expected);

        invoke_on_all_policies<4>()(test_short_reduce(), in.cbegin(), in.cend(), init, expected);
        invoke_on_all_policies<5>()(test_short_reduce_init(), in.cbegin(), in.cend(), init, expected);
    }
}

int
main()
{
    // Test for popular types
    test_long_form(42, ::std::plus<std::int32_t>(), [](std::int32_t x) { return x; });
    test_long_form(42.0, ::std::plus<float64_t>(), [](float64_t x) { return x; });

#if !TEST_DPCPP_BACKEND_PRESENT
    // Test for strict types
    // Creation of temporary buffer from const iterators requires default ctor of Number
    // TODO: fix it
    test_long_form<Number>(Number(42, OddTag()), Add(OddTag()), [](std::int32_t x) { return Number(x, OddTag()); });
#endif

    // Short forms are just facade for long forms, so just test with a single type.
    test_short_forms();

    return done();
}
