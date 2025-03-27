// -*- C++ -*-
//===-- transform_unary.pass.cpp ------------------------------------------===//
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

// Tests for transform

#include "support/test_config.h"

#include _PSTL_TEST_HEADER(execution)
#include _PSTL_TEST_HEADER(algorithm)

#include "support/utils.h"

using namespace TestUtils;

struct test_one_policy
{
    template <typename Policy, typename InputIterator, typename OutputIterator, typename UnaryOp>
    void
    operator()(Policy&& exec, InputIterator first, InputIterator last, OutputIterator out_first,
               OutputIterator out_last, UnaryOp op)
    {
        auto orr = ::std::transform(exec, first, last, out_first, op);
        EXPECT_TRUE(out_last == orr, "transform returned wrong iterator");
        check_and_reset(first, last, out_first);
    }

private:
    template <typename InputIterator, typename OutputIterator>
    void
    check_and_reset(InputIterator first, InputIterator last, OutputIterator out_first)
    {
        typename ::std::iterator_traits<OutputIterator>::difference_type k = 0;
        for (; first != last; ++first, ++out_first, ++k)
        {
            // check
            const auto expected = get_expected(*first);
            auto& actual = get_actual(*out_first);
            EXPECT_EQ(expected, actual, "wrong value in output sequence");
            // reset
            actual = k % 7 != 4 ? 7 * k - 5 : 0;
        }
    }
    template <typename InputT>
    auto
    get_expected(InputT val)
    {
        return 1 - val;
    }
    template <typename OutputT>
    auto&
    get_actual(OutputT& val)
    {
        return val;
    }

    template <typename T>
    auto
    get_expected(oneapi::dpl::__internal::tuple<T&>&& t)
    {
        return 1 - std::get<0>(t);
    }
    template <typename T>
    auto&
    get_actual(oneapi::dpl::__internal::tuple<T&>&& t)
    {
        return std::get<0>(t);
    }
};

template <::std::size_t CallNumber, typename Tin, typename Tout, typename _Op = Complement<Tin, Tout>,
          typename _IteratorAdapter = _Identity>
void
test()
{
    for (size_t n : TestUtils::get_pattern_for_test_sizes())
    {
        Sequence<Tin> in(n, [](std::int32_t k) { return k % 5 != 1 ? 3 * k - 7 : 0; });

        Sequence<Tout> out(n);

        _IteratorAdapter adap;
        invoke_on_all_policies<CallNumber>()(test_one_policy(), adap(in.begin()), adap(in.end()), adap(out.begin()),
                                             adap(out.end()), _Op{});

#if !ONEDPL_FPGA_DEVICE
        invoke_on_all_policies<CallNumber + 1>()(test_one_policy(), adap(in.cbegin()), adap(in.cend()),
                                                 adap(out.begin()), adap(out.end()), _Op{});
#endif
    }
}

template <typename T>
struct test_non_const
{
    template <typename Policy, typename InputIterator, typename OutputInterator>
    void
    operator()(Policy&& exec, InputIterator input_iter, OutputInterator out_iter)
    {
        transform(exec, input_iter, input_iter, out_iter, non_const(::std::negate<T>()));
    }
};

int
main()
{
    test<0, std::int32_t, std::int32_t>();
    test<10, std::int32_t, float32_t>();
    test<20, std::uint16_t, float32_t>();
    test<30, float64_t, float64_t>();

    test_algo_basic_double<std::int32_t>(run_for_rnd_fw<test_non_const<std::int32_t>>());
    test_algo_basic_double<std::int64_t>(run_for_rnd_fw<test_non_const<std::int32_t>>());

    //test case for zip iterator
    test<40, std::int32_t, std::int32_t, ComplementZip, _ZipIteratorAdapter>();

    return done();
}
