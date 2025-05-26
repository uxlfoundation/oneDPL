// -*- C++ -*-
//===-- transform_reduce.pass.cpp -----------------------------------------===//
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

// Equal for all types
template <typename T>
static bool
Equal(T x, T y)
{
    return x == y;
}

// Functor for xor-operation for modeling binary operations in inner_product
class XOR
{
  public:
    template <typename T>
    T
    operator()(const T& left, const T& right) const
    {
        return left ^ right;
    }
};

// Model of User-defined class
class MyClass
{
  public:
    std::int32_t my_field;
    MyClass() { my_field = 0; }
    MyClass(std::int32_t in) { my_field = in; }

    friend MyClass
    operator+(const MyClass& x, const MyClass& y)
    {
        return MyClass(x.my_field + y.my_field);
    }
    friend MyClass
    operator-(const MyClass& x)
    {
        return MyClass(-x.my_field);
    }
    friend MyClass operator*(const MyClass& x, const MyClass& y) { return MyClass(x.my_field * y.my_field); }
    bool
    operator==(const MyClass& in) const
    {
        return my_field == in.my_field;
    }
};

template <typename T>
void
CheckResults(const T& expected, const T& in, const char* msg)
{
    EXPECT_TRUE(Equal(expected, in), msg);
}

// We need to check correctness only for "int" (for example) except cases
// if we have "floating-point type"-specialization
void
CheckResults(const float32_t& /* expected */, const float32_t& /* in */, const char* /* msg */)
{
}

// Test for different types and operations with different iterators
template <typename Type>
struct test_3_iters_default_ops
{
    template <typename Policy, typename InputIterator1, typename InputIterator2, typename T>
    void
    operator()(Policy&& exec, InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, InputIterator2 /* last2 */,
               T init)
    {
        auto expectedB = ::std::inner_product(first1, last1, first2, init);
        T resRA = ::std::transform_reduce(exec, first1, last1, first2, init);
        CheckResults(expectedB, resRA, "wrong result with tranform_reduce (3 iterators, default predicates)");
    }
};

template <typename Type>
struct test_3_iters_custom_ops
{
    template <typename Policy, typename InputIterator1, typename InputIterator2, typename T, typename BinaryOperation1,
              typename BinaryOperation2>
    void
    operator()(Policy&& exec, InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, InputIterator2 /* last2 */,
               T init, BinaryOperation1 opB1, BinaryOperation2 opB2)
    {
        auto expectedB = ::std::inner_product(first1, last1, first2, init, opB1, opB2);
        T resRA = ::std::transform_reduce(exec, first1, last1, first2, init, opB1, opB2);
        CheckResults(expectedB, resRA, "wrong result with tranform_reduce (3 iterators, custom predicates)");
    }
};

template  <typename Type>
struct test_2_iters
{
    template <typename Policy, typename InputIterator1, typename T, typename BinaryOperation,
              typename UnaryOp>
    void
    operator()(Policy&& exec, InputIterator1 first1, InputIterator1 last1, T init, BinaryOperation opB, UnaryOp opU)
    {
        auto expectedU = transform_reduce_serial(first1, last1, init, opB, opU);
        T resRA = ::std::transform_reduce(exec, first1, last1, init, opB, opU);
        CheckResults(expectedU, resRA, "wrong result with tranform_reduce (2 iterators)");
    }
};

template <typename T, typename BinaryOperation1, typename BinaryOperation2, typename UnaryOp, typename Initializer>
void
test_by_type(T init, BinaryOperation1 opB1, BinaryOperation2 opB2, UnaryOp opU, Initializer initObj)
{
    ::std::size_t maxSize = 100000;
    Sequence<T> in1(maxSize, initObj);
    Sequence<T> in2(maxSize, initObj);

    for (::std::size_t n = 0; n < maxSize; n = n < 16 ? n + 1 : size_t(3.1415 * n))
    {
        invoke_on_all_policies<0>()(test_3_iters_custom_ops<T>(), in1.begin(), in1.begin() + n,
                                    in2.begin(), in2.begin() + n, init, opB1, opB2);
        invoke_on_all_policies<1>()(test_2_iters<T>(), in1.begin(), in1.begin() + n, init, opB1, opU);
#if !ONEDPL_FPGA_DEVICE
        invoke_on_all_policies<2>()(test_3_iters_default_ops<T>(), in1.begin(), in1.begin() + n,
                                    in2.begin(), in2.begin() + n, init);

        invoke_on_all_policies<3>()(test_3_iters_custom_ops<T>(), in1.cbegin(), in1.cbegin() + n,
                                    in2.cbegin(), in2.cbegin() + n, init, opB1, opB2);
        invoke_on_all_policies<4>()(test_2_iters<T>(), in1.cbegin(), in1.cbegin() + n, init, opB1, opU);
#endif
    }
}

int
main()
{
    test_by_type<std::int32_t>(42, ::std::plus<std::int32_t>(), ::std::multiplies<std::int32_t>(), ::std::negate<std::int32_t>(),
                          [](::std::size_t) -> std::int32_t { return std::int32_t(rand() % 1000); });
    test_by_type<std::int64_t>(0, [](const std::int64_t& a, const std::int64_t& b) -> std::int64_t { return a | b; }, XOR(),
                          [](const std::int64_t& x) -> std::int64_t { return x * 2; },
                          [](::std::size_t) -> std::int64_t { return std::int64_t(rand() % 1000); });
    test_by_type<float32_t>(1.0f, ::std::multiplies<float32_t>(),
                            [](const float32_t& a, const float32_t& b) -> float32_t { return a + b; },
                            [](const float32_t& x) -> float32_t { return x + 2; },
                            [](::std::size_t) -> float32_t { return rand() % 1000; });
    test_by_type<MyClass>(MyClass(), ::std::plus<MyClass>(), ::std::multiplies<MyClass>(),
        [](const MyClass& x) { return MyClass(-x.my_field); },
        [](::std::size_t) -> MyClass { return MyClass(rand() % 1000); });
    return done();
}
