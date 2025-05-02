// -*- C++ -*-
//===-- find_first_of.pass.cpp --------------------------------------------===//
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
struct test_find_first_of
{
    template <typename ExecutionPolicy, typename Iterator1, typename Iterator2>
    void
    operator()(ExecutionPolicy&& exec, Iterator1 b, Iterator1 e, Iterator2 bsub, Iterator2 esub)
    {
        using namespace std;
        Iterator1 expected = find_first_of(b, e, bsub, esub);
        Iterator1 actual = find_first_of(exec, b, e, bsub, esub);
        EXPECT_TRUE(actual == expected, "wrong return result from find_first_of");
    }
};

template <typename T>
struct test_find_first_of_predicate
{
    template <typename ExecutionPolicy, typename Iterator1, typename Iterator2, typename Predicate>
    void
    operator()(ExecutionPolicy&& exec, Iterator1 b, Iterator1 e, Iterator2 bsub, Iterator2 esub, Predicate pred)
    {
        using namespace std;
        Iterator1 expected = find_first_of(b, e, bsub, esub, pred);
        Iterator1 actual = find_first_of(exec, b, e, bsub, esub, pred);
        EXPECT_TRUE(actual == expected, "wrong return result from find_first_of with a predicate");
    }
};

template <typename T, typename Predicate>
void
test(Predicate pred)
{

    const ::std::size_t max_n1 = 1000;
    const ::std::size_t max_n2 = (max_n1 * 10) / 8;
    Sequence<T> in1(max_n1, [](::std::size_t) { return T(1); });
    Sequence<T> in2(max_n2, [](::std::size_t) { return T(0); });
    for (::std::size_t n1 = 0; n1 <= max_n1; n1 = n1 <= 16 ? n1 + 1 : size_t(3.1415 * n1))
    {
        ::std::size_t sub_n[] = {0, 1, n1 / 3, n1, (n1 * 10) / 8};
        for (const auto n2 : sub_n)
        {
            invoke_on_all_policies<0>()(test_find_first_of<T>(), in1.begin(), in1.begin() + n1, in2.begin(),
                                        in2.begin() + n2);
            invoke_on_all_policies<1>()(test_find_first_of_predicate<T>(), in1.begin(), in1.begin() + n1, in2.begin(),
                                        in2.begin() + n2, pred);

            in2[n2 / 2] = T(1);
#if !TEST_DPCPP_BACKEND_PRESENT
            invoke_on_all_policies<2>()(test_find_first_of<T>(), in1.cbegin(), in1.cbegin() + n1, in2.data(),
                                        in2.data() + n2);
            invoke_on_all_policies<3>()(test_find_first_of_predicate<T>(), in1.cbegin(), in1.cbegin() + n1, in2.data(),
                                        in2.data() + n2, pred);
#else
#if !ONEDPL_FPGA_DEVICE
            invoke_on_all_policies<2>()(test_find_first_of<T>(), in1.cbegin(), in1.cbegin() + n1, in2.begin(),
                                        in2.begin() + n2);
            invoke_on_all_policies<3>()(test_find_first_of_predicate<T>(), in1.cbegin(), in1.cbegin() + n1, in2.begin(),
                                        in2.begin() + n2, pred);
#endif
#endif
            if (n2 >= 3)
            {
                in2[2 * n2 / 3] = T(1);
                invoke_on_all_policies<4>()(test_find_first_of<T>(), in1.cbegin(), in1.cbegin() + n1, in2.begin(),
                                            in2.begin() + n2);
                invoke_on_all_policies<5>()(test_find_first_of_predicate<T>(), in1.cbegin(), in1.cbegin() + n1,
                                            in2.begin(), in2.begin() + n2, pred);
                in2[2 * n2 / 3] = T(0);
            }
            in2[n2 / 2] = T(0);
        }
    }
    invoke_on_all_policies<6>()(test_find_first_of<T>(), in1.begin(), in1.begin() + max_n1 / 10, in1.begin(),
                                in1.begin() + max_n1 / 10);
    invoke_on_all_policies<7>()(test_find_first_of_predicate<T>(), in1.begin(), in1.begin() + max_n1 / 10, in1.begin(),
                                in1.begin() + max_n1 / 10, pred);
}

template <typename T>
struct test_non_const
{
    template <typename Policy, typename FirstIterator, typename SecondInterator>
    void
    operator()(Policy&& exec, FirstIterator first_iter, SecondInterator second_iter)
    {
        find_first_of(exec, first_iter, first_iter, second_iter, second_iter, non_const(::std::equal_to<T>()));
    }
};

int
main()
{
    test<std::int32_t>(::std::equal_to<std::int32_t>());
#if !ONEDPL_FPGA_DEVICE
    test<std::uint16_t>(::std::not_equal_to<std::uint16_t>());
#endif
    test<float64_t>([](const float64_t x, const float64_t y) { return x * x == y * y; });

    test_algo_basic_double<std::int32_t>(run_for_rnd_fw<test_non_const<std::int32_t>>());

    return done();
}
