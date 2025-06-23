// -*- C++ -*-
//===-- mismatch.pass.cpp -------------------------------------------------===//
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

template <typename Type>
struct test_mismatch
{
    template <typename Policy, typename Iterator1, typename Iterator2>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2)
    {
        using namespace std;
        typedef typename iterator_traits<Iterator1>::value_type T;
        {
            const auto expected = ::std::mismatch(first1, last1, first2, ::std::equal_to<T>());
            const auto res4 = mismatch(std::forward<Policy>(exec), first1, last1, first2);
            EXPECT_EQ(expected, res4, "wrong return result from mismatch");
        }
    }
    template <typename Policy, typename Iterator1, typename Iterator2>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 last2)
    {
        using namespace std;
        typedef typename iterator_traits<Iterator1>::value_type T;
        {
            const auto expected = mismatch(oneapi::dpl::execution::seq, first1, last1, first2, last2, ::std::equal_to<T>());
            const auto res2 = mismatch(std::forward<Policy>(exec), first1, last1, first2, last2);
            EXPECT_EQ(expected, res2, "wrong return result from mismatch");
        }
    }
};

template <typename Type>
struct test_mismatch_predicate
{
    template <typename Policy, typename Iterator1, typename Iterator2>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2)
    {
        using namespace std;
        typedef typename iterator_traits<Iterator1>::value_type T;
        {
            const auto expected = ::std::mismatch(first1, last1, first2, ::std::equal_to<T>());
            const auto res3 = mismatch(std::forward<Policy>(exec), first1, last1, first2, std::equal_to<T>());
            EXPECT_EQ(expected, res3, "wrong return result from mismatch with predicate");
        }
    }
    template <typename Policy, typename Iterator1, typename Iterator2>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 last2)
    {
        using namespace std;
        typedef typename iterator_traits<Iterator1>::value_type T;
        {
            const auto expected = mismatch(oneapi::dpl::execution::seq, first1, last1, first2, last2, ::std::equal_to<T>());
            const auto res1 = mismatch(std::forward<Policy>(exec), first1, last1, first2, last2, std::equal_to<T>());
            EXPECT_EQ(expected, res1, "wrong return result from mismatch with predicate");
        }
    }
};

template <typename T>
void
test_mismatch_by_type()
{
    using namespace std;
    for (size_t size = 0; size <= 100000; size = size <= 16 ? size + 1 : size_t(3.1415 * size))
    {
        const T val = T(-1);
        Sequence<T> in(size, [](size_t v) -> T { return T(v % 100); });
        {
            Sequence<T> in2(in);
            invoke_on_all_policies<0>()(test_mismatch<T>(), in.begin(), in.end(), in2.begin(), in2.end());
#if !ONEDPL_FPGA_DEVICE
            invoke_on_all_policies<1>()(test_mismatch<T>(), in.begin(), in.end(), in2.begin());
            invoke_on_all_policies<2>()(test_mismatch_predicate<T>(), in.begin(), in.end(), in2.begin(), in2.end());
#endif
            invoke_on_all_policies<3>()(test_mismatch_predicate<T>(), in.begin(), in.end(), in2.begin());

            const size_t min_size = 3;
            if (size > min_size)
            {
                const size_t idx_for_1 = size / min_size;
                in[idx_for_1] = val, in[idx_for_1 + 1] = val, in[idx_for_1 + 2] = val;
                invoke_on_all_policies<4>()(test_mismatch<T>(), in.begin(), in.end(), in2.begin(), in2.end());
                invoke_on_all_policies<5>()(test_mismatch<T>(), in.begin(), in.end(), in2.begin());
                invoke_on_all_policies<6>()(test_mismatch_predicate<T>(), in.begin(), in.end(), in2.begin(),
                                            in2.end());
                invoke_on_all_policies<7>()(test_mismatch_predicate<T>(), in.begin(), in.end(), in2.begin());
            }

            const size_t idx_for_2 = 500;
            if (size >= idx_for_2 - 1)
            {
                in2[size / idx_for_2] = val;
                invoke_on_all_policies<8>()(test_mismatch<T>(), in.cbegin(), in.cend(), in2.cbegin(), in2.cend());
#if !ONEDPL_FPGA_DEVICE
                invoke_on_all_policies<9>()(test_mismatch<T>(), in.cbegin(), in.cend(), in2.cbegin());
                invoke_on_all_policies<10>()(test_mismatch_predicate<T>(), in.cbegin(), in.cend(), in2.cbegin(),
                                             in2.cend());
#endif
                invoke_on_all_policies<11>()(test_mismatch_predicate<T>(), in.cbegin(), in.cend(), in2.cbegin());
            }
        }
        {
            Sequence<T> in2(100, [](size_t v) -> T { return T(v); });
            invoke_on_all_policies<12>()(test_mismatch<T>(), in2.begin(), in2.end(), in.begin(), in.end());
            invoke_on_all_policies<13>()(test_mismatch_predicate<T>(), in2.begin(), in2.end(), in.begin(), in.end());
            //  We can't call ::std::mismatch with semantic below when size of second sequence less than size of first sequence
            if (in2.size() <= in.size())
            {
                invoke_on_all_policies<14>()(test_mismatch<T>(), in2.begin(), in2.end(), in.begin());
                invoke_on_all_policies<15>()(test_mismatch_predicate<T>(), in2.begin(), in2.end(), in.begin());
            }

            const size_t idx = 97;
            in2[idx] = val;
            in2[idx + 1] = val;
            invoke_on_all_policies<16>()(test_mismatch<T>(), in.cbegin(), in.cend(), in2.cbegin(), in2.cend());
            invoke_on_all_policies<17>()(test_mismatch_predicate<T>(), in.cbegin(), in.cend(), in2.cbegin(),
                                         in2.cend());

            if (in.size() <= in2.size())
            {
                invoke_on_all_policies<18>()(test_mismatch<T>(), in.cbegin(), in.cend(), in2.cbegin());
                invoke_on_all_policies<19>()(test_mismatch_predicate<T>(), in.cbegin(), in.cend(), in2.cbegin());
            }
        }
        {
            Sequence<T> in2({});
            invoke_on_all_policies<20>()(test_mismatch<T>(), in2.begin(), in2.end(), in.begin(), in.end());
#if !ONEDPL_FPGA_DEVICE
            invoke_on_all_policies<21>()(test_mismatch_predicate<T>(), in2.begin(), in2.end(), in.begin(), in.end());
            invoke_on_all_policies<22>()(test_mismatch<T>(), in.cbegin(), in.cend(), in2.cbegin(), in2.cend());
#endif
            invoke_on_all_policies<23>()(test_mismatch_predicate<T>(), in.cbegin(), in.cend(), in2.cbegin(),
                                         in2.cend());

            if (in.size() == 0)
            {
                invoke_on_all_policies<24>()(test_mismatch<T>(), in.cbegin(), in.cend(), in2.cbegin());
                invoke_on_all_policies<25>()(test_mismatch_predicate<T>(), in.cbegin(), in.cend(), in2.cbegin());
            }
        }
    }
}

template <typename T>
struct test_non_const
{
    template <typename Policy, typename FirstIterator, typename SecondInterator>
    void
    operator()(Policy&& exec, FirstIterator first_iter, SecondInterator second_iter)
    {
        mismatch(std::forward<Policy>(exec), first_iter, first_iter, second_iter, second_iter, non_const(std::less<T>()));
    }
};

int
main()
{

    test_mismatch_by_type<std::int32_t>();
#if !ONEDPL_FPGA_DEVICE
    test_mismatch_by_type<float64_t>();
#endif

#if !TEST_DPCPP_BACKEND_PRESENT
    test_mismatch_by_type<Wrapper<std::int32_t>>();
#endif
    test_algo_basic_double<std::int32_t>(run_for_rnd_fw<test_non_const<std::int32_t>>());

    return done();
}
