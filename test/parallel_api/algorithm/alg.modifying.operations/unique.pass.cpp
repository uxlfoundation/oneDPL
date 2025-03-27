// -*- C++ -*-
//===-- unique.pass.cpp ---------------------------------------------------===//
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

// Test for unique
#include "support/test_config.h"

#include _PSTL_TEST_HEADER(execution)
#include _PSTL_TEST_HEADER(algorithm)

#include "support/utils.h"

using namespace TestUtils;

template <typename T>
struct run_unique
{
    template <typename Policy, typename ForwardIt, typename Generator>
    void
    operator()(Policy&& exec, ForwardIt first1, ForwardIt last1, ForwardIt first2, ForwardIt last2, Generator generator)
    {
        using namespace std;

        // Preparation
        fill_data(first1, last1, generator);
        fill_data(first2, last2, generator);

        ForwardIt i = unique(first1, last1);
        ForwardIt k = unique(exec, first2, last2);

        auto n = ::std::distance(first1, i);
        EXPECT_TRUE(::std::distance(first2, k) == n, "wrong return value from unique without predicate");
        EXPECT_EQ_N(first1, first2, n, "wrong effect from unique without predicate");
    }
};

template <typename T>
struct run_unique_predicate
{
    template <typename Policy, typename ForwardIt, typename BinaryPred, typename Generator>
    void
    operator()(Policy&& exec, ForwardIt first1, ForwardIt last1, ForwardIt first2, ForwardIt last2, BinaryPred pred,
               Generator generator)
    {
        using namespace std;

        // Preparation
        fill_data(first1, last1, generator);
        fill_data(first2, last2, generator);

        ForwardIt i = unique(first1, last1, pred);
        ForwardIt k = unique(exec, first2, last2, pred);

        auto n = ::std::distance(first1, i);
        EXPECT_TRUE(::std::distance(first2, k) == n, "wrong return value from unique with predicate");
        EXPECT_EQ_N(first1, first2, n, "wrong effect from unique with predicate");
    }
};

template <typename T, typename Generator, typename Predicate>
void
test(Generator generator, Predicate pred)
{
    const ::std::size_t max_size = 1000000;
    Sequence<T> in(max_size, [](size_t v) { return T(v); });
    Sequence<T> exp(max_size, [](size_t v) { return T(v); });

    for (size_t n = 0; n <= max_size; n = n <= 16 ? n + 1 : size_t(3.1415 * n))
    {
        invoke_on_all_policies<>()(run_unique<T>(), exp.begin(), exp.begin() + n, in.begin(), in.begin() + n,
                                   generator);
        invoke_on_all_policies<>()(run_unique_predicate<T>(), exp.begin(), exp.begin() + n, in.begin(),
                                   in.begin() + n, pred, generator);
    }
}

template <typename T>
struct LocalWrapper
{
    T my_val;

    explicit LocalWrapper(T k) : my_val(k) {}
    LocalWrapper(LocalWrapper&& input) : my_val(::std::move(input.my_val)) {}
    LocalWrapper&
    operator=(LocalWrapper&& input)
    {
        my_val = ::std::move(input.my_val);
        return *this;
    }
    friend bool
    operator==(const LocalWrapper<T>& x, const LocalWrapper<T>& y)
    {
        return x.my_val == y.my_val;
    }
};

template <typename T>
struct test_non_const
{
    template <typename Policy, typename Iterator>
    void
    operator()(Policy&& exec, Iterator iter)
    {
        unique(exec, iter, iter, non_const(::std::equal_to<T>()));
    }
};

int
main()
{
    test<std::int32_t>([](size_t j) { return j / 3; },
                  [](const std::int32_t& val1, const std::int32_t& val2) { return val1 * val1 == val2 * val2; });
#if !ONEDPL_FPGA_DEVICE
    test<float64_t>([](size_t) { return float64_t(1); },
                    [](const float64_t& val1, const float64_t& val2) { return val1 != val2; });
#endif

#if !TEST_DPCPP_BACKEND_PRESENT
    test<LocalWrapper<std::uint32_t>>([](size_t j) { return LocalWrapper<std::uint32_t>(j); },
                                 [](const LocalWrapper<std::uint32_t>& val1, const LocalWrapper<std::uint32_t>& val2) {
                                     return val1.my_val != val2.my_val;
                                 });
    test<MemoryChecker>(
        [](::std::size_t idx){ return MemoryChecker{::std::int32_t(idx / 3)}; },
        [](const MemoryChecker& val1, const MemoryChecker& val2){ return val1.value() == val2.value(); });
    EXPECT_TRUE(MemoryChecker::alive_objects() == 0, "wrong effect from unique: number of ctor and dtor calls is not equal");
#endif
    test_algo_basic_single<std::int32_t>(run_for_rnd_fw<test_non_const<std::int32_t>>());

    return done();
}
