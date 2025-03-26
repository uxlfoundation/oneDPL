// -*- C++ -*-
//===-- transform_binary.pass.cpp -----------------------------------------===//
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

struct test_one_policy
{
    template <typename Policy, typename InputIterator1, typename InputIterator2, typename OutputIterator,
              typename BinaryOp>
    void
    operator()(Policy&& exec, InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, InputIterator2 /* last2 */,
               OutputIterator out_first, OutputIterator /* out_last */, BinaryOp op)
    {
        ::std::transform(exec, first1, last1, first2, out_first, op);
        check_and_reset(first1, last1, first2, out_first);
    }

    template <typename InputIterator1, typename InputIterator2, typename OutputIterator>
    void
    check_and_reset(InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, OutputIterator out_first)
    {
        typename ::std::iterator_traits<OutputIterator>::difference_type k = 0;
        for (; first1 != last1; ++first1, ++first2, ++out_first, ++k)
        {
            // check
            using Out = decltype(get_out_type(*out_first));
            const auto expected = get_expected<Out>(*first1, *first2);
            auto& actual = get_actual(*out_first);
            EXPECT_EQ(expected, actual, "wrong value in output sequence");
            // reset
            actual = k % 7 != 4 ? 7 * k + 5 : 0;
        }
    }
    template <typename Out, typename InputT1, typename InputT2>
    Out
    get_expected(InputT1 val1, InputT2 val2)
    {
        return Out(1.5) + val1 - val2;
    }
    template <typename OutputT>
    auto&
    get_actual(OutputT& val)
    {
        return val;
    }

    template <typename Out, typename T1, typename T2>
    auto
    get_expected(oneapi::dpl::__internal::tuple<T1&>&& t1, oneapi::dpl::__internal::tuple<T2&>&& t2)
    {
        return Out(1.5) + std::get<0>(t1) - std::get<0>(t2);
    }

    template <typename T>
    auto&
    get_actual(oneapi::dpl::__internal::tuple<T&>&& t)
    {
        return std::get<0>(t);
    }

    template <typename T>
    constexpr auto
    get_out_type(T val)
    {
        return val;
    }

    template <typename T>
    constexpr auto
    get_out_type(oneapi::dpl::__internal::tuple<T&>&& t)
    {
        return std::get<0>(t);;
    }
};

template <::std::size_t CallNumber, typename In1, typename In2, typename Out, typename Predicate,
          typename _IteratorAdapter = _Identity>
void
test(Predicate pred, _IteratorAdapter adap = {})
{
    // Testing is restricted for debug build + OpenMP backend as without optimization the compiler generates
    // very slow code leading to test timeouts.

    const auto test_sizes = TestUtils::get_pattern_for_test_sizes();
    size_t max_n =
#if PSTL_USE_DEBUG && ONEDPL_USE_OPENMP_BACKEND
        10000;
#else
        test_sizes.back();
#endif
    for (size_t n : test_sizes)
    {
        if (n <= max_n)
        {
            Sequence<In1> in1(n, [](size_t k) { return k % 5 != 1 ? In1(3 * k + 7) : 0; });
            Sequence<In2> in2(n, [](size_t k) { return k % 7 != 2 ? In2(5 * k + 5) : 0; });

            Sequence<Out> out(n, [](size_t) { return -1; });

            invoke_on_all_policies<CallNumber>()(test_one_policy(), adap(in1.begin()), adap(in1.end()), adap(in2.begin()),
                                                 adap(in2.end()), adap(out.begin()), adap(out.end()), pred);
            invoke_on_all_policies<CallNumber + 1>()(test_one_policy(), adap(in1.cbegin()), adap(in1.cend()),
                                                     adap(in2.cbegin()), adap(in2.cend()), adap(out.begin()),
                                                     adap(out.end()), pred);
        }
    }
}

template <typename T>
struct test_non_const
{
    template <typename Policy, typename InputIterator, typename OutputInterator>
    void
    operator()(Policy&& exec, InputIterator input_iter, OutputInterator out_iter)
    {
        InputIterator input_iter2 = input_iter;
        transform(exec, input_iter, input_iter, input_iter2, out_iter, non_const(::std::plus<T>()));
    }
};

int
main()
{
    //const operator()
    test<0, std::int32_t, std::int32_t, std::int32_t>(TheOperation<std::int32_t, std::int32_t, std::int32_t>(1));
    test<10, float32_t, float32_t, float32_t>(TheOperation<float32_t, float32_t, float32_t>(1.5));
    //non-const operator()
#if !TEST_DPCPP_BACKEND_PRESENT
    test<20, std::int32_t, float32_t, float32_t>(non_const(TheOperation<std::int32_t, float32_t, float32_t>(1.5)));
    test<30, std::int64_t, float64_t, float32_t>(non_const(TheOperation<std::int64_t, float64_t, float32_t>(1.5)));
#endif
    test<40, std::int32_t, float64_t, std::int32_t>([](const std::int32_t& x, const float64_t& y) { return std::int32_t(std::int32_t(1.5) + x - y); });

    test_algo_basic_double<std::int16_t>(run_for_rnd_fw<test_non_const<std::int16_t>>());

    //test case for zip iterator
    test<50, std::int32_t, std::int32_t, std::int32_t>(TheOperationZip<std::int32_t>(1), _ZipIteratorAdapter{});

    test<60, std::uint16_t, std::uint16_t, std::int32_t>(TheOperation<std::uint16_t, std::uint16_t, std::int32_t>(1));

    return done();
}
