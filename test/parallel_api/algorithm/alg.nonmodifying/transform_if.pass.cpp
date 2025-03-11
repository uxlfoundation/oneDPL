// -*- C++ -*-
//===-- transform_if.pass.cpp ----------------------------------------------------===//
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
#include _PSTL_TEST_HEADER(algorithm)

#include "support/utils.h"

using namespace TestUtils;

template <typename T1, typename T2>
struct mutable_negate_first
{
    T1
    operator()(const T1& a, const T2&) //explicitly not const
    {
        return -a;
    }
};

template <typename T>
struct mutable_negate
{
    T
    operator()(const T& a) //explicitly not const
    {
        return -a;
    }
};

template <typename T1, typename T2>
struct mutable_check_mask_second
{
    bool
    operator()(const T1&, const T2& b) //explicitly not const
    {
        return b == 1;
    }
};

template <typename T>
struct mutable_check_mod3_is_0
{
    bool
    operator()(const T& a) //explicitly not const
    {
        return (a % 3) == 0;
    }
};

template <typename T1>
struct test_transform_if_binary
{
    template <typename Policy, typename InputIterator1, typename InputIterator2, typename OutputIterator, typename Size>
    void
    operator()(Policy&& exec, InputIterator1 first, InputIterator1 last, InputIterator2 mask,
               InputIterator2 /*mask_end*/, OutputIterator result_begin, OutputIterator result_end, Size n,
               ::std::int64_t init_val)
    {
        using in_value_type1 = typename ::std::iterator_traits<InputIterator1>::value_type;
        using in_value_type2 = typename ::std::iterator_traits<InputIterator2>::value_type;
        using out_value_type = typename ::std::iterator_traits<OutputIterator>::value_type;
        // call transform_if
        oneapi::dpl::transform_if(exec, first, last, mask, result_begin,
                                  mutable_negate_first<in_value_type1, in_value_type2>{},
                                  mutable_check_mask_second<in_value_type1, in_value_type2>{});

        //calculate expected
        std::vector<out_value_type> expected(n);
        auto in_iter = first;
        auto mask_iter = mask;
        auto expected_iter = expected.begin();
        for (; in_iter != last; in_iter++, mask_iter++, expected_iter++)
        {
            *expected_iter = *mask_iter == 1 ? -(*in_iter) : out_value_type(init_val);
        }

        EXPECT_EQ_N(expected.begin(), result_begin, n, "wrong effect from transform_if binary");

        // reset output elements to init value as output_range value is passed through where predicate is false
        ::std::fill(result_begin, result_end, out_value_type(init_val));
    }
};

template <typename T1>
struct test_transform_if_unary
{
    template <typename Policy, typename InputIterator1, typename OutputIterator, typename Size>
    void
    operator()(Policy&& exec, InputIterator1 first, InputIterator1 last, OutputIterator result_begin,
               OutputIterator result_end, Size n, ::std::int64_t init_val)
    {
        using in_value_type = typename ::std::iterator_traits<InputIterator1>::value_type;
        using out_value_type = typename ::std::iterator_traits<OutputIterator>::value_type;

        // call transform_if
        oneapi::dpl::transform_if(exec, first, last, result_begin, mutable_negate<in_value_type>{},
                                  mutable_check_mod3_is_0<in_value_type>{});

        //calculate expected
        std::vector<out_value_type> expected(n);
        auto expected_iter = expected.begin();
        auto in_iter = first;
        for (; in_iter != last; in_iter++, expected_iter++)
        {
            *expected_iter = *in_iter % 3 == 0 ? -(*in_iter) : out_value_type(init_val);
        }

        EXPECT_EQ_N(expected.begin(), result_begin, n, "wrong effect from transform_if unary");

        // reset output elements to init value as output_range value is passed through where predicate is false
        ::std::fill(result_begin, result_end, out_value_type(init_val));
    }
};

template <typename T1>
struct test_transform_if_unary_inplace
{
    template <typename Policy, typename InputIterator1, typename OutputIterator, typename Size>
    void
    operator()(Policy&& exec, InputIterator1 first, InputIterator1 last, OutputIterator result_begin,
               OutputIterator result_end, Size n)
    {
        using in_value_type = typename ::std::iterator_traits<InputIterator1>::value_type;

        // Start with a fresh input to the inplace test
        std::copy(first, last, result_begin);

        // call transform_if inplace
        oneapi::dpl::transform_if(exec, result_begin, result_end, result_begin, mutable_negate<in_value_type>{},
                                  mutable_check_mod3_is_0<in_value_type>{});

        //calculate expected
        std::vector<in_value_type> expected(n);
        auto expected_iter = expected.begin();
        auto in_iter = first;
        for (; in_iter != last; in_iter++, expected_iter++)
        {
            *expected_iter = *in_iter % 3 == 0 ? -(*in_iter) : *in_iter;
        }

        EXPECT_EQ_N(expected.begin(), result_begin, n, "wrong effect from transform_if inplace unary");
    }
};

template <typename _Type>
void
test()
{
    const ::std::int64_t init_val = 999;
    for (size_t n : TestUtils::get_pattern_for_test_sizes())
    {
        {
            Sequence<_Type> in1(n, [=](size_t k) { return (3 * k) % std::numeric_limits<_Type>::max(); });
            Sequence<_Type> in2(n, [=](size_t k) { return k % 2 == 0 ? 1 : 0; });

            Sequence<_Type> out(n, [=](size_t) { return init_val; });

            invoke_on_all_policies<0>()(test_transform_if_binary<_Type>(), in1.begin(), in1.end(), in2.begin(),
                                        in2.end(), out.begin(), out.end(), n, init_val);
#if !ONEDPL_FPGA_DEVICE
            invoke_on_all_policies<1>()(test_transform_if_binary<_Type>(), in1.cbegin(), in1.cend(), in2.cbegin(),
                                        in2.cend(), out.begin(), out.end(), n, init_val);
#endif
        }
        {
            Sequence<_Type> in1(n, [=](size_t k) { return k % std::numeric_limits<_Type>::max(); });
            Sequence<_Type> out(n, [=](size_t) { return init_val; });

            invoke_on_all_policies<2>()(test_transform_if_unary<_Type>(), in1.begin(), in1.end(), out.begin(),
                                        out.end(), n, init_val);
#if !ONEDPL_FPGA_DEVICE
            invoke_on_all_policies<3>()(test_transform_if_unary<_Type>(), in1.cbegin(), in1.cend(), out.begin(),
                                        out.end(), n, init_val);
#endif
        }
    }
}

template <typename _Type>
void
test_inplace()
{
    for (size_t n : TestUtils::get_pattern_for_test_sizes())
    {
        {
            Sequence<_Type> in1(n, [=](size_t k) { return k % std::numeric_limits<_Type>::max(); });
            Sequence<_Type> out(n, [=](size_t) { return 0; });

            invoke_on_all_policies<4>()(test_transform_if_unary_inplace<_Type>(), in1.begin(), in1.end(), out.begin(),
                                        out.end(), n);
        }
    }
}

int
main()
{
    test<std::int8_t>();
    test<std::int16_t>();
    test<std::int32_t>();
    test<std::int64_t>();

    test_inplace<std::int8_t>();
    test_inplace<std::int16_t>();
    test_inplace<std::int32_t>();
    test_inplace<std::int64_t>();

    return done();
}
