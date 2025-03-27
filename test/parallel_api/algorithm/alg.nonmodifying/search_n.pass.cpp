// -*- C++ -*-
//===-- search_n.pass.cpp -------------------------------------------------===//
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
struct test_search_n
{
    template <typename ExecutionPolicy, typename Iterator, typename Size, typename T, typename Predicate>
    void
    operator()(ExecutionPolicy&& exec, Iterator b, Iterator e, Size count, const T& value, Predicate pred)
    {
        using namespace std;
        auto expected = search_n(b, e, count, value, pred);
        auto actual = search_n(exec, b, e, count, value);
        EXPECT_TRUE(actual == expected, "wrong return result from search_n");
    }
};

template <typename Type>
struct test_search_n_predicate
{
    template <typename ExecutionPolicy, typename Iterator, typename Size, typename T, typename Predicate>
    void
    operator()(ExecutionPolicy&& exec, Iterator b, Iterator e, Size count, const T& value, Predicate pred)
    {
        using namespace std;
        auto expected = search_n(b, e, count, value, pred);
        auto actual = search_n(exec, b, e, count, value, pred);
        EXPECT_TRUE(actual == expected, "wrong return result from search_n with a predicate");
    }
};

template <typename T>
void
test()
{

    const ::std::size_t max_n1 = 100000;
    const T value = T(1);
    for (::std::size_t n1 = 0; n1 <= max_n1; n1 = n1 <= 16 ? n1 + 1 : size_t(3.1415 * n1))
    {
        ::std::size_t sub_n[] = {0, 1, 3, n1, (n1 * 10) / 8};
        ::std::size_t res[] = {0, 1, n1 / 2, n1};
        for (auto n2 : sub_n)
        {
            for (auto r : res)
            {
                Sequence<T> in(n1, [](::std::size_t) { return T(0); });
                ::std::size_t i = r, isub = 0;
                for (; i < n1 && isub < n2; ++i, ++isub)
                    in[i] = value;

                invoke_on_all_policies<0>()(test_search_n<T>(), in.begin(), in.begin() + n1, n2, value,
                                            ::std::equal_to<T>());
                invoke_on_all_policies<1>()(test_search_n_predicate<T>(), in.begin(), in.begin() + n1, n2, value,
                                            ::std::equal_to<T>());
#if !ONEDPL_FPGA_DEVICE
                invoke_on_all_policies<2>()(test_search_n<T>(), in.cbegin(), in.cbegin() + n1, n2, value,
                                            ::std::equal_to<T>());
                invoke_on_all_policies<3>()(test_search_n_predicate<T>(), in.cbegin(), in.cbegin() + n1, n2, value,
                                            ::std::equal_to<T>());
#endif
            }
        }
    }
}

template <typename T>
struct test_non_const
{
    template <typename Policy, typename Iterator>
    void
    operator()(Policy&& exec, Iterator iter)
    {
        search_n(exec, iter, iter, 0, T(0), non_const(::std::equal_to<T>()));
    }
};

int
main()
{
    test<std::int32_t>();
#if !ONEDPL_FPGA_DEVICE
    test<std::uint16_t>();
#endif
    test<float64_t>();
    test<bool>();

    test_algo_basic_single<std::int32_t>(run_for_rnd_fw<test_non_const<std::int32_t>>());

    return done();
}
