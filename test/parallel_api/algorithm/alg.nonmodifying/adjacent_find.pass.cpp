// -*- C++ -*-
//===-- adjacent_find.pass.cpp --------------------------------------------===//
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

template <typename T>
struct test_adjacent_find
{
    template <typename Policy, typename Iterator>
    void
    operator()(Policy&& exec, Iterator first, Iterator last)
    {
        using namespace std;

        auto k = ::std::adjacent_find(first, last);
        auto i = adjacent_find(std::forward<Policy>(exec), first, last);
        EXPECT_TRUE(i == k, "wrong return value from adjacent_find without predicate");
    }
};

template <typename T>
struct test_adjacent_find_predicate
{
    template <typename Policy, typename Iterator, typename Pred>
    void
    operator()(Policy&& exec, Iterator first, Iterator last, Pred pred)
    {
        using namespace std;

        auto k = ::std::adjacent_find(first, last, pred);
        auto i = adjacent_find(std::forward<Policy>(exec), first, last, pred);
        EXPECT_TRUE(i == k, "wrong return value from adjacent_find with predicate");
    }
};

template <typename T>
struct CustomPred
{
    bool operator()(T x, T y) const
    {
        return (x - y) * (x - y) == 4;
    }
};

template <typename T>
void
test_adjacent_find_by_type()
{
    CustomPred<T> custom_pred;
    size_t counts[] = {2, 3, 500};
    for (std::int32_t c = 0; c < const_size(counts); ++c)
    {

        for (std::int32_t e = 0; e < (counts[c] >= 64 ? 64 : (counts[c] == 2 ? 1 : 2)); ++e)
        {
            Sequence<T> in(counts[c], [](std::int32_t v) -> T { return T(v); }); //fill 0...n
            in[e] = in[e + 1]+2;                                         //make an adjacent pair

            auto i = ::std::adjacent_find(in.cbegin(), in.cend(), custom_pred);
            EXPECT_TRUE(i == in.cbegin() + e, "::std::adjacent_find returned wrong result");

            invoke_on_all_policies<0>()(test_adjacent_find<T>(), in.begin(), in.end());
            invoke_on_all_policies<1>()(test_adjacent_find_predicate<T>(), in.begin(), in.end(), custom_pred);
#if !ONEDPL_FPGA_DEVICE
            invoke_on_all_policies<2>()(test_adjacent_find<T>(), in.cbegin(), in.cend());
            invoke_on_all_policies<3>()(test_adjacent_find_predicate<T>(), in.cbegin(), in.cend(), custom_pred);
#endif
        }
    }

    //special cases: size=0, size=1;
    for (std::int32_t expect = 0; expect < 1; ++expect)
    {
        Sequence<T> in(expect, [](std::int32_t v) -> T { return T(v); }); //fill 0...n
        auto i = ::std::adjacent_find(in.cbegin(), in.cend(), custom_pred);
        EXPECT_TRUE(i == in.cbegin() + expect, "::std::adjacent_find returned wrong result");

        invoke_on_all_policies<4>()(test_adjacent_find<T>(), in.begin(), in.end());
        invoke_on_all_policies<5>()(test_adjacent_find_predicate<T>(), in.begin(), in.end(), custom_pred);
#if !ONEDPL_FPGA_DEVICE
        invoke_on_all_policies<6>()(test_adjacent_find<T>(), in.cbegin(), in.cend());
        invoke_on_all_policies<7>()(test_adjacent_find_predicate<T>(), in.cbegin(), in.cend(), custom_pred);
#endif
    }

    //special cases:
    Sequence<T> a1 = { 1, 5, 3, 1, 7, 8, 9 };
    invoke_on_all_policies<8>()(test_adjacent_find<T>(), a1.begin(), a1.end());
    invoke_on_all_policies<9>()(test_adjacent_find<T>(), a1.begin() + 1, a1.end());
    invoke_on_all_policies<10>()(test_adjacent_find_predicate<T>(), a1.begin(), a1.end(), custom_pred);
    invoke_on_all_policies<11>()(test_adjacent_find_predicate<T>(), a1.begin() + 1, a1.end(), custom_pred);

#if !ONEDPL_FPGA_DEVICE
    invoke_on_all_policies<12>()(test_adjacent_find<T>(), a1.cbegin(), a1.cend());
    invoke_on_all_policies<13>()(test_adjacent_find<T>(), a1.cbegin() + 1, a1.cend());
    invoke_on_all_policies<14>()(test_adjacent_find_predicate<T>(), a1.cbegin(), a1.cend(), custom_pred);
    invoke_on_all_policies<15>()(test_adjacent_find_predicate<T>(), a1.cbegin() + 1, a1.cend(), custom_pred);
#endif

    Sequence<T> a2 = { 5, 6, 7, 9, 9, 11 };
    invoke_on_all_policies<16>()(test_adjacent_find<T>(), a2.begin(), a2.end());
    invoke_on_all_policies<17>()(test_adjacent_find<T>(), a2.begin(), a2.end() - 1);
    invoke_on_all_policies<18>()(test_adjacent_find_predicate<T>(), a2.begin(), a2.end(), custom_pred);
    invoke_on_all_policies<19>()(test_adjacent_find_predicate<T>(), a2.begin(), a2.end() - 1, custom_pred);

#if !ONEDPL_FPGA_DEVICE
    invoke_on_all_policies<20>()(test_adjacent_find<T>(), a2.cbegin(), a2.cend());
    invoke_on_all_policies<21>()(test_adjacent_find<T>(), a2.cbegin(), a2.cend() - 1);
    invoke_on_all_policies<22>()(test_adjacent_find_predicate<T>(), a2.cbegin(), a2.cend(), custom_pred);
    invoke_on_all_policies<23>()(test_adjacent_find_predicate<T>(), a2.cbegin(), a2.cend() - 1, custom_pred);
#endif

    Sequence<T> a3 = { 5, 6, 6, 7, 8, 9, 9, 9, 11 };
    invoke_on_all_policies<24>()(test_adjacent_find<T>(), a3.begin(), a3.end());
    invoke_on_all_policies<25>()(test_adjacent_find_predicate<T>(), a3.begin(), a3.end(), custom_pred);
#if !ONEDPL_FPGA_DEVICE
    invoke_on_all_policies<26>()(test_adjacent_find<T>(), a3.cbegin(), a3.cend());
    invoke_on_all_policies<27>()(test_adjacent_find_predicate<T>(), a3.cbegin(), a3.cend(), custom_pred);
#endif
}

template <typename T>
struct test_non_const
{
    template <typename Policy, typename Iterator>
    void
    operator()(Policy&& exec, Iterator iter)
    {
        adjacent_find(std::forward<Policy>(exec), iter, iter, non_const(std::equal_to<T>()));
    }
};

int
main()
{

#if !ONEDPL_FPGA_DEVICE
    test_adjacent_find_by_type<std::int32_t>();
#endif
    test_adjacent_find_by_type<float64_t>();
    test_algo_basic_single<std::int32_t>(run_for_rnd_bi<test_non_const<std::int32_t>>());

    return done();
}
