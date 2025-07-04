// -*- C++ -*-
//===-- lexicographical_compare.pass.cpp ----------------------------------===//
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

#include <iostream>

using namespace TestUtils;

template <typename T>
struct test_one_policy
{

    template <typename ExecutionPolicy, typename Iterator1, typename Iterator2, typename Predicate>
    void
    operator()(ExecutionPolicy&& exec, Iterator1 begin1, Iterator1 end1, Iterator2 begin2, Iterator2 end2,
               Predicate pred)
    {
        const bool expected = ::std::lexicographical_compare(begin1, end1, begin2, end2, pred);
        const bool actual = std::lexicographical_compare(std::forward<ExecutionPolicy>(exec), begin1, end1, begin2, end2, pred);
        EXPECT_EQ(expected, actual, "wrong return result from lexicographical compare with predicate");
    }

    template <typename ExecutionPolicy, typename Iterator1, typename Iterator2>
    void
    operator()(ExecutionPolicy&& exec, Iterator1 begin1, Iterator1 end1, Iterator2 begin2, Iterator2 end2)
    {
        const bool expected = ::std::lexicographical_compare(begin1, end1, begin2, end2);
        const bool actual = std::lexicographical_compare(std::forward<ExecutionPolicy>(exec), begin1, end1, begin2, end2);
        EXPECT_EQ(expected, actual, "wrong return result from lexicographical compare without predicate");
    }
};

template <typename T1, typename T2, typename Predicate>
void
test(Predicate pred)
{

    const ::std::size_t max_size = 1000000;
    Sequence<T1> in1(max_size, [](::std::size_t k) { return T1(k); });
    Sequence<T2> in2(2 * max_size, [](::std::size_t k) { return T2(k); });

    ::std::size_t n2;

    // Test case: Call algorithm's version without predicate.
    invoke_on_all_policies<0>()(test_one_policy<T1>(), in1.cbegin(), in1.cbegin() + max_size,
                                in2.cbegin() + 3 * max_size / 10, in2.cbegin() + 5 * max_size / 10);

    // Test case: If one range is a prefix of another, the shorter range is lexicographically less than the other.
    ::std::size_t max_size2 = max_size / 10;
    invoke_on_all_policies<1>()(test_one_policy<T1>(), in1.begin(), in1.begin() + max_size, in2.cbegin(),
                                in2.cbegin() + max_size2, pred);
    invoke_on_all_policies<2>()(test_one_policy<T1>(), in1.begin(), in1.begin() + max_size, in2.begin() + max_size2,
                                in2.begin() + 3 * max_size2, pred);

    // Test case: If one range is a prefix of another, the shorter range is lexicographically less than the other.
    max_size2 = 2 * max_size;
    invoke_on_all_policies<3>()(test_one_policy<T1>(), in1.cbegin(), in1.cbegin() + max_size, in2.begin(),
                                in2.begin() + max_size2, pred);

    for (::std::size_t n1 = 0; n1 <= max_size; n1 = n1 <= 16 ? n1 + 1 : ::std::size_t(3.1415 * n1))
    {
        // Test case: If two ranges have equivalent elements and are of the same length, then the ranges are lexicographically equal.
        n2 = n1;
        invoke_on_all_policies<4>()(test_one_policy<T1>(), in1.begin(), in1.begin() + n1, in2.begin(),
                                    in2.begin() + n2, pred);

        n2 = n1;
        // Test case: two ranges have different elements and are of the same length (second sequence less than first)
        ::std::size_t ind = n1 / 2;
        in2[ind] = T2(-1);
        invoke_on_all_policies<5>()(test_one_policy<T1>(), in1.begin(), in1.begin() + n1, in2.begin(),
                                    in2.begin() + n2, pred);
        in2[ind] = T2(ind);

        // Test case: two ranges have different elements and are of the same length (first sequence less than second)
        ind = n1 / 5;
        in1[ind] = T1(-1);
        invoke_on_all_policies<6>()(test_one_policy<T1>(), in1.begin(), in1.begin() + n1, in2.cbegin(),
                                    in2.cbegin() + n2, pred);
        in1[ind] = T1(ind);
    }
}

template <typename Predicate>
void
test_string(Predicate pred)
{
    const std::size_t max_size = 1000000;
    ::std::string in1 = "";
    ::std::string in2 = "";
    for (::std::size_t n1 = 0; n1 <= max_size; ++n1)
    {
        in1 += n1;
    }

    for (std::size_t n1 = 0; n1 <= 2 * max_size; ++n1)
    {
        in2 += n1;
    }

    ::std::size_t n2;

    for (::std::size_t n1 = 0; n1 < in1.size(); n1 = n1 <= 16 ? n1 + 1 : ::std::size_t(3.1415 * n1))
    {
        // Test case: If two ranges have equivalent elements and are of the same length, then the ranges are lexicographically equal.
        n2 = n1;
        invoke_on_all_policies<7>()(test_one_policy<Predicate>(), in1.begin(), in1.begin() + n1, in2.begin(),
                                    in2.begin() + n2, pred);

        n2 = n1;
        // Test case: two ranges have different elements and are of the same length (second sequence less than first)
        in2[n1 / 2] = 'a';
        invoke_on_all_policies<8>()(test_one_policy<Predicate>(), in1.begin(), in1.begin() + n1, in2.begin(),
                                    in2.begin() + n2, pred);

        // Test case: two ranges have different elements and are of the same length (first sequence less than second)
        in1[n1 / 5] = 'a';
        invoke_on_all_policies<9>()(test_one_policy<Predicate>(), in1.begin(), in1.begin() + n1, in2.cbegin(),
                                    in2.cbegin() + n2, pred);
    }
    invoke_on_all_policies<10>()(test_one_policy<Predicate>(), in1.cbegin(), in1.cbegin() + max_size,
                                 in2.cbegin() + 3 * max_size / 10, in2.cbegin() + 5 * max_size / 10);
}

template <typename T>
struct LocalWrapper
{
    explicit LocalWrapper(::std::size_t k) : my_val(k) {}
    bool
    operator<(const LocalWrapper<T>& w) const
    {
        return my_val < w.my_val;
    }

  private:
    T my_val;
};

template <typename T>
struct test_non_const
{
    template <typename Policy, typename FirstIterator, typename SecondInterator>
    void
    operator()(Policy&& exec, FirstIterator first_iter, SecondInterator second_iter)
    {
        lexicographical_compare(std::forward<Policy>(exec), first_iter, first_iter, second_iter, second_iter, non_const(std::less<T>()));
    }
};

int
main()
{
    test<std::uint16_t, float64_t>(::std::less<float64_t>());

#if !ONEDPL_FPGA_DEVICE
    test<float32_t, std::int32_t>(::std::greater<float32_t>());
#if !_PSTL_ICC_18_TEST_EARLY_EXIT_AVX_RELEASE_BROKEN
    test<float64_t, std::int32_t>([](const float64_t x, const std::int32_t y) { return x * x < y * y; });
#endif
#endif

#if !TEST_DPCPP_BACKEND_PRESENT
    test<LocalWrapper<std::int32_t>, LocalWrapper<std::int32_t>>(
        [](const LocalWrapper<std::int32_t>& x, const LocalWrapper<std::int32_t>& y) { return x < y; });

    test_string([](const char x, const char y) { return x < y; });
#endif

    test_algo_basic_double<std::int32_t>(run_for_rnd_fw<test_non_const<std::int32_t>>());

    return done();
}
