// -*- C++ -*-
//===-- is_sorted.pass.cpp ------------------------------------------------===//
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

#if !defined(_PSTL_TEST_IS_SORTED) && !defined(_PSTL_TEST_IS_SORTED_UNTIL)
#    define _PSTL_TEST_IS_SORTED
#    define _PSTL_TEST_IS_SORTED_UNTIL
#endif

template <typename Type>
struct test_is_sorted
{
    template <typename Policy, typename Iterator>
    void
    operator()(Policy&& exec, Iterator first, Iterator last)
    {
        using namespace std;

        //try random-access iterator
        bool exam = is_sorted(first, last);
        bool res = is_sorted(std::forward<Policy>(exec), first, last);
        EXPECT_TRUE(exam == res, "is_sorted wrong result for random-access iterator");
    }
};

template <typename Type>
struct test_is_sorted_predicate
{
    template <typename Policy, typename Iterator>
    void
    operator()(Policy&& exec, Iterator first, Iterator last)
    {
        using namespace std;
        typedef typename ::std::iterator_traits<Iterator>::value_type T;

        //try random-access iterator with a predicate
        bool exam = is_sorted(first, last, ::std::less<T>());
        bool res = is_sorted(std::forward<Policy>(exec), first, last, std::less<T>());
        EXPECT_TRUE(exam == res, "is_sorted wrong result for random-access iterator");
    }
};

template <typename Type>
struct test_is_sorted_until
{
    template <typename Policy, typename Iterator>
    void
    operator()(Policy&& exec, Iterator first, Iterator last)
    {
        using namespace std;

        //try random-access iterator
        auto iexam = is_sorted_until(first, last);
        auto ires = is_sorted_until(std::forward<Policy>(exec), first, last);
        EXPECT_TRUE(iexam == ires, "is_sorted_until wrong result for random-access iterator");
    }
};

template <typename Type>
struct test_is_sorted_until_predicate
{
    template <typename Policy, typename Iterator>
    void
    operator()(Policy&& exec, Iterator first, Iterator last)
    {
        using namespace std;
        typedef typename ::std::iterator_traits<Iterator>::value_type T;

        //try random-access iterator with a predicate
        auto iexam = is_sorted_until(first, last, ::std::less<T>());
        auto ires = is_sorted_until(std::forward<Policy>(exec), first, last, std::less<T>());
        EXPECT_TRUE(iexam == ires, "is_sorted_until with predicate wrong result for random-access iterator");
    }
};

template <typename T>
void
test_is_sorted_by_type()
{
    const ::std::size_t max_size = 1000000;

    for (::std::size_t n1 = 1; n1 <= max_size; n1 = n1 <= 16 ? n1 + 1 : ::std::size_t(3.1415 * n1))
    {
        Sequence<T> in(n1, [](::std::size_t v) -> T { return T(v); }); //fill 0..n
#ifdef _PSTL_TEST_IS_SORTED
        invoke_on_all_policies<0>()(test_is_sorted<T>(), in.begin(), in.end());
        invoke_on_all_policies<1>()(test_is_sorted_predicate<T>(), in.begin(), in.end());
#    if !ONEDPL_FPGA_DEVICE
        invoke_on_all_policies<4>()(test_is_sorted<T>(), in.cbegin(), in.cend());
        invoke_on_all_policies<5>()(test_is_sorted_predicate<T>(), in.cbegin(), in.cend());
#    endif
#endif

#ifdef _PSTL_TEST_IS_SORTED_UNTIL
        invoke_on_all_policies<2>()(test_is_sorted_until<T>(), in.begin(), in.end());
        invoke_on_all_policies<3>()(test_is_sorted_until_predicate<T>(), in.begin(), in.end());
#    if !ONEDPL_FPGA_DEVICE
        invoke_on_all_policies<6>()(test_is_sorted_until<T>(), in.cbegin(), in.cend());
        invoke_on_all_policies<7>()(test_is_sorted_until_predicate<T>(), in.cbegin(), in.cend());
#    endif
#endif

        in[in.size() / 2] = -1;
#ifdef _PSTL_TEST_IS_SORTED
        invoke_on_all_policies<8>()(test_is_sorted<T>(), in.begin(), in.end());
        invoke_on_all_policies<9>()(test_is_sorted_predicate<T>(), in.begin(), in.end());
#    if !ONEDPL_FPGA_DEVICE
        invoke_on_all_policies<12>()(test_is_sorted<T>(), in.cbegin(), in.cend());
        invoke_on_all_policies<13>()(test_is_sorted_predicate<T>(), in.cbegin(), in.cend());
#    endif
#endif

#ifdef _PSTL_TEST_IS_SORTED_UNTIL
        invoke_on_all_policies<10>()(test_is_sorted_until<T>(), in.begin(), in.end());
        invoke_on_all_policies<11>()(test_is_sorted_until_predicate<T>(), in.begin(), in.end());
#    if !ONEDPL_FPGA_DEVICE
        invoke_on_all_policies<14>()(test_is_sorted_until<T>(), in.cbegin(), in.cend());
        invoke_on_all_policies<15>()(test_is_sorted_until_predicate<T>(), in.cbegin(), in.cend());
#    endif
#endif

        in[in.size() / 10] = -1;
#ifdef _PSTL_TEST_IS_SORTED
        invoke_on_all_policies<16>()(test_is_sorted<T>(), in.begin(), in.end());
        invoke_on_all_policies<17>()(test_is_sorted_predicate<T>(), in.begin(), in.end());
#    if !ONEDPL_FPGA_DEVICE
        invoke_on_all_policies<20>()(test_is_sorted<T>(), in.cbegin(), in.cend());
        invoke_on_all_policies<21>()(test_is_sorted_predicate<T>(), in.cbegin(), in.cend());
#    endif
#endif

#ifdef _PSTL_TEST_IS_SORTED_UNTIL
        invoke_on_all_policies<18>()(test_is_sorted_until<T>(), in.begin(), in.end());
        invoke_on_all_policies<19>()(test_is_sorted_until_predicate<T>(), in.begin(), in.end());
#    if !ONEDPL_FPGA_DEVICE
        invoke_on_all_policies<22>()(test_is_sorted_until<T>(), in.cbegin(), in.cend());
        invoke_on_all_policies<23>()(test_is_sorted_until_predicate<T>(), in.cbegin(), in.cend());
#    endif
#endif
    }

    //an empty container
    Sequence<T> in0(0);
#ifdef _PSTL_TEST_IS_SORTED
    invoke_on_all_policies<24>()(test_is_sorted<T>(), in0.begin(), in0.end());
    invoke_on_all_policies<25>()(test_is_sorted_predicate<T>(), in0.begin(), in0.end());
#    if !ONEDPL_FPGA_DEVICE
    invoke_on_all_policies<28>()(test_is_sorted<T>(), in0.cbegin(), in0.cend());
    invoke_on_all_policies<29>()(test_is_sorted_predicate<T>(), in0.cbegin(), in0.cend());
#    endif
#endif

#ifdef _PSTL_TEST_IS_SORTED_UNTIL
    invoke_on_all_policies<26>()(test_is_sorted_until<T>(), in0.begin(), in0.end());
    invoke_on_all_policies<27>()(test_is_sorted_until_predicate<T>(), in0.begin(), in0.end());
#    if !ONEDPL_FPGA_DEVICE
    invoke_on_all_policies<30>()(test_is_sorted_until<T>(), in0.cbegin(), in0.cend());
    invoke_on_all_policies<31>()(test_is_sorted_until_predicate<T>(), in0.cbegin(), in0.cend());
#    endif
#endif

    //non-descending order
    Sequence<T> in1(9, [](size_t) -> T { return T(0); });
#ifdef _PSTL_TEST_IS_SORTED
    invoke_on_all_policies<32>()(test_is_sorted<T>(), in1.begin(), in1.end());
    invoke_on_all_policies<33>()(test_is_sorted_predicate<T>(), in1.begin(), in1.end());
#    if !ONEDPL_FPGA_DEVICE
    invoke_on_all_policies<36>()(test_is_sorted<T>(), in1.cbegin(), in1.cend());
    invoke_on_all_policies<37>()(test_is_sorted_predicate<T>(), in1.cbegin(), in1.cend());
#    endif
#endif

#ifdef _PSTL_TEST_IS_SORTED_UNTIL
    invoke_on_all_policies<34>()(test_is_sorted_until<T>(), in1.begin(), in1.end());
    invoke_on_all_policies<35>()(test_is_sorted_until_predicate<T>(), in1.begin(), in1.end());
#    if !ONEDPL_FPGA_DEVICE
    invoke_on_all_policies<38>()(test_is_sorted_until<T>(), in1.cbegin(), in1.cend());
    invoke_on_all_policies<39>()(test_is_sorted_until_predicate<T>(), in1.cbegin(), in1.cend());
#    endif
#endif
}

template <typename T>
struct test_non_const_is_sorted
{
    template <typename Policy, typename Iterator>
    void
    operator()(Policy&& exec, Iterator iter)
    {
        is_sorted(std::forward<Policy>(exec), iter, iter, std::less<T>());
    }
};

template <typename T>
struct test_non_const_is_sorted_until
{
    template <typename Policy, typename Iterator>
    void
    operator()(Policy&& exec, Iterator iter)
    {
        is_sorted_until(std::forward<Policy>(exec), iter, iter, std::less<T>());
    }
};

int
main()
{
    test_is_sorted_by_type<std::int32_t>();
#if !ONEDPL_FPGA_DEVICE
    test_is_sorted_by_type<float64_t>();
#endif

#if !TEST_DPCPP_BACKEND_PRESENT
    test_is_sorted_by_type<Wrapper<std::int32_t>>();
#endif
#ifdef _PSTL_TEST_IS_SORTED
    test_algo_basic_single<std::int32_t>(run_for_rnd_fw<test_non_const_is_sorted<std::int32_t>>());
#endif
#ifdef _PSTL_TEST_IS_SORTED_UNTIL
    test_algo_basic_single<std::int32_t>(run_for_rnd_fw<test_non_const_is_sorted_until<std::int32_t>>());
#endif

    return done();
}
