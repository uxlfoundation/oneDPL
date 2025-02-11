// -*- C++ -*-
//===-- generate.pass.cpp -------------------------------------------------===//
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

#include <atomic>

#if  !defined(_PSTL_TEST_GENERATE) && !defined(_PSTL_TEST_GENERATE_N)
#define _PSTL_TEST_GENERATE
#define _PSTL_TEST_GENERATE_N
#endif

using namespace TestUtils;

template <typename T>
struct Generator_count
{
    const T def_val = T(-1);
    T
    operator()() const
    {
        return def_val;
    }
    T
    default_value() const
    {
        return def_val;
    }
};

template <typename T>
struct test_generate
{
    template <typename Policy, typename Iterator, typename Size>
    void
    operator()(Policy&& exec, Iterator first, Iterator last, Size n)
    {
        using namespace std;
        Generator_count<T> g;
        generate(exec, first, last, g);
        EXPECT_TRUE(::std::count(first, last, g.default_value()) == n, "generate wrong result for generate");
        ::std::fill(first, last, T(0));
    }
};

template <typename T>
struct test_generate_n
{
    template <typename Policy, typename Iterator, typename Size>
    void
    operator()(Policy&& exec, Iterator first, Iterator /* last */, Size n)
    {
        using namespace std;

        Generator_count<T> g;
        const auto m = n / 2;
        auto gen_last = generate_n(exec, first, m, g);
        EXPECT_TRUE(::std::count(first, gen_last, g.default_value()) == m && gen_last == ::std::next(first, m),
                    "generate_n wrong result for generate_n");
        ::std::fill(first, gen_last, T(0));
    }
};

template <typename T>
void
test_generate_by_type()
{
    for (size_t n : TestUtils::get_pattern_for_test_sizes())
    {
        Sequence<T> in(n, [](size_t) -> T { return T(0); }); //fill by zero

#ifdef _PSTL_TEST_GENERATE
        invoke_on_all_policies<>()(test_generate<T>(), in.begin(), in.end(), in.size());
#endif
#ifdef _PSTL_TEST_GENERATE_N
        invoke_on_all_policies<>()(test_generate_n<T>(), in.begin(), in.end(), in.size());
#endif

    }
}

template <typename T>
struct test_non_const_generate
{
    template <typename Policy, typename Iterator>
    void
    operator()(Policy&& exec, Iterator iter)
    {
        auto gen = []() { return T(0); };
        generate(exec, iter, iter, non_const(gen));
    }
};

template <typename T>
struct test_non_const_generate_n
{
    template <typename Policy, typename Iterator>
    void
    operator()(Policy&& exec, Iterator iter)
    {
        auto gen = []() { return T(0); };
        generate_n(exec, iter, 0, non_const(gen));
    }
};

int
main()
{
    test_generate_by_type<std::uint8_t>();
    test_generate_by_type<std::int32_t>();
    test_generate_by_type<float64_t>();


#ifdef _PSTL_TEST_GENERATE
	test_algo_basic_single<std::int16_t>(run_for_rnd_fw<test_non_const_generate<std::int16_t>>());
#endif
#ifdef _PSTL_TEST_GENERATE_N
    test_algo_basic_single<std::int16_t>(run_for_rnd_fw<test_non_const_generate_n<std::int16_t>>());
#endif

    return done();
}
