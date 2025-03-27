// -*- C++ -*-
//===-- is_partitioned.pass.cpp -------------------------------------------===//
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
struct test_is_partitioned
{
    template <typename ExecutionPolicy, typename Iterator1, typename Predicate>
    void
    operator()(ExecutionPolicy&& exec, Iterator1 begin1, Iterator1 end1, Predicate pred)
    {
        const bool expected = ::std::is_partitioned(begin1, end1, pred);
        const bool actual = ::std::is_partitioned(exec, begin1, end1, pred);
        EXPECT_TRUE(actual == expected, "wrong return result from is_partitioned");
    }
};

template <typename T, typename Predicate>
void
test(Predicate pred)
{

    const ::std::size_t max_n = 1000000;
    Sequence<T> in(max_n, [](::std::size_t k) { return T(k); });

    for (::std::size_t n1 = 0; n1 <= max_n; n1 = n1 <= 16 ? n1 + 1 : ::std::size_t(3.1415 * n1))
    {
        invoke_on_all_policies<0>()(test_is_partitioned<T>(), in.begin(), in.begin() + n1, pred);
        ::std::partition(in.begin(), in.begin() + n1, pred);
        invoke_on_all_policies<1>()(test_is_partitioned<T>(), in.cbegin(), in.cbegin() + n1, pred);
    }
}

template <typename T>
struct LocalWrapper
{
    explicit LocalWrapper(::std::size_t k) : my_val(k) {}

  private:
    T my_val;
};

struct test_non_const
{
    template <typename Policy, typename Iterator>
    void
    operator()(Policy&& exec, Iterator iter)
    {
        auto is_even = [&](float64_t v) {
            std::uint32_t i = (std::uint32_t)v;
            return i % 2 == 0;
        };
        is_partitioned(exec, iter, iter, non_const(is_even));
    }
};

int
main()
{
    test<float64_t>([](const float64_t x) { return x < 0; });
    test<std::int32_t>([](const std::int32_t x) { return x > 1000; });
    test<std::uint16_t>([](const std::uint16_t x) { return x % 5 < 3; });

#if !TEST_DPCPP_BACKEND_PRESENT && !_PSTL_ICC_18_TEST_EARLY_EXIT_MONOTONIC_RELEASE_BROKEN &&               \
    !_PSTL_ICC_19_TEST_IS_PARTITIONED_RELEASE_BROKEN
    test<LocalWrapper<float64_t>>([](const LocalWrapper<float64_t>&) { return true; });
    test_algo_basic_single<std::int32_t>(run_for_rnd_fw<test_non_const>());
#endif

    return done();
}
