// -*- C++ -*-
//===-- uninitialized_construct.pass.cpp ----------------------------------===//
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

// Tests for uninitialized_default_consruct, uninitialized_default_consruct_n,
//           uninitialized_value_consruct,   uninitialized_value_consruct_n

#include "support/test_config.h"

#include _PSTL_TEST_HEADER(execution)
#include _PSTL_TEST_HEADER(memory)

#include "support/utils.h"

#if  !defined(_PSTL_TEST_UNITIALIZED_DEFAULT_CONSTRUCT) && !defined(_PSTL_TEST_UNITIALIZED_DEFAULT_CONSTRUCT_N) &&\
     !defined(_PSTL_TEST_UNITIALIZED_VALUE_CONSTRUCT) && !defined(_PSTL_TEST_UNITIALIZED_VALUE_CONSTRUCT_N)
#define _PSTL_TEST_UNITIALIZED_DEFAULT_CONSTRUCT
#define _PSTL_TEST_UNITIALIZED_DEFAULT_CONSTRUCT_N
#define _PSTL_TEST_UNITIALIZED_VALUE_CONSTRUCT
#define _PSTL_TEST_UNITIALIZED_VALUE_CONSTRUCT_N
#endif

using namespace TestUtils;

// function of checking correctness for uninitialized.construct.value
template <typename T, typename Iterator>
bool
IsCheckValueCorrectness(Iterator begin, Iterator end)
{
    for (; begin != end; ++begin)
    {
        if (*begin != T())
        {
            return false;
        }
    }
    return true;
}

template <typename Type>
struct test_uninit_default_construct
{
    template <typename Policy, typename Iterator>
    void
    operator()(Policy&& exec, Iterator begin, Iterator end, size_t n, /*is_trivial<T>=*/::std::false_type)
    {
        using T = typename std::iterator_traits<Iterator>::value_type;
        // it needs for cleaning memory that was filled by default constructors in unique_ptr<T[]> p(new T[n])
        // and for cleaning memory after last calling of uninitialized_value_construct_n.
        // It is important for non-trivial types
        ::std::destroy_n(oneapi::dpl::execution::seq, begin, n);
        T::SetCount(0);

        std::uninitialized_default_construct(std::forward<Policy>(exec), begin, end);
        EXPECT_EQ(n, T::Count(), "wrong uninitialized_default_construct");

    }

    template <typename Policy, typename Iterator>
    void
    operator()(Policy&& exec, Iterator begin, Iterator end, size_t /* n */, /*is_trivial<T>=*/::std::true_type)
    {
        std::uninitialized_default_construct(std::forward<Policy>(exec), begin, end);
    }
};

template <typename Type>
struct test_uninit_default_construct_n
{
    template <typename Policy, typename Iterator>
    void
    operator()(Policy&& exec, Iterator begin, Iterator /* end */, size_t n, /*is_trivial<T>=*/::std::false_type)
    {
        using T = typename std::iterator_traits<Iterator>::value_type;
        // it needs for cleaning memory that was filled by default constructors in unique_ptr<T[]> p(new T[n])
        // and for cleaning memory after last calling of uninitialized_value_construct_n.
        // It is important for non-trivial types
        ::std::destroy_n(oneapi::dpl::execution::seq, begin, n);
        T::SetCount(0);

        std::uninitialized_default_construct_n(std::forward<Policy>(exec), begin, n);
        EXPECT_EQ(n, T::Count(), "wrong uninitialized_default_construct_n");
    }

    template <typename Policy, typename Iterator>
    void
    operator()(Policy&& exec, Iterator begin, Iterator /* end */, size_t n, /*is_trivial<T>=*/::std::true_type)
    {
        std::uninitialized_default_construct_n(std::forward<Policy>(exec), begin, n);
    }
};

template <typename Type>
struct test_uninit_value_construct
{
    template <typename Policy, typename Iterator>
    void
    operator()(Policy&& exec, Iterator begin, Iterator end, size_t n, /*is_trivial<T>=*/::std::false_type)
    {
        using T = typename std::iterator_traits<Iterator>::value_type;
        // it needs for cleaning memory that was filled by default constructors in unique_ptr<T[]> p(new T[n])
        // and for cleaning memory after last calling of uninitialized_value_construct_n.
        // It is important for non-trivial types
        ::std::destroy_n(oneapi::dpl::execution::seq, begin, n);
        T::SetCount(0);

        std::uninitialized_value_construct(std::forward<Policy>(exec), begin, end);
        EXPECT_EQ(n, T::Count(), "wrong uninitialized_value_construct");
    }

    template <typename Policy, typename Iterator>
    void
    operator()(Policy&& exec, Iterator begin, Iterator end, size_t /* n */, /*is_trivial<T>=*/::std::true_type)
    {
        using T = typename std::iterator_traits<Iterator>::value_type;

        std::uninitialized_value_construct(std::forward<Policy>(exec), begin, end);
        EXPECT_TRUE(IsCheckValueCorrectness<T>(begin, end), "wrong uninitialized_value_construct");
    }
};

template <typename Type>
struct test_uninit_value_construct_n
{
    template <typename Policy, typename Iterator>
    void
    operator()(Policy&& exec, Iterator begin, Iterator /* end */, size_t n, /*is_trivial<T>=*/::std::false_type)
    {
        using T = typename std::iterator_traits<Iterator>::value_type;
        // it needs for cleaning memory that was filled by default constructors in unique_ptr<T[]> p(new T[n])
        // and for cleaning memory after last calling of uninitialized_value_construct_n.
        // It is important for non-trivial types
        ::std::destroy_n(oneapi::dpl::execution::seq, begin, n);
        T::SetCount(0);

        std::uninitialized_value_construct_n(std::forward<Policy>(exec), begin, n);
        EXPECT_EQ(n, T::Count(), "wrong uninitialized_value_construct_n");
    }

    template <typename Policy, typename Iterator>
    void
    operator()(Policy&& exec, Iterator begin, Iterator end, size_t n, /*is_trivial<T>=*/::std::true_type)
    {
        using T = typename std::iterator_traits<Iterator>::value_type;

        std::uninitialized_value_construct_n(std::forward<Policy>(exec), begin, n);
        EXPECT_TRUE(IsCheckValueCorrectness<T>(begin, end), "wrong uninitialized_value_construct_n");
    }
};

template <typename T>
void
test_uninit_construct_by_type()
{
    ::std::size_t N = 100000;
    for (size_t n = 0; n <= N; n = n <= 16 ? n + 1 : size_t(3.1415 * n))
    {
#if !TEST_DPCPP_BACKEND_PRESENT
        ::std::unique_ptr<T[]> p(new T[n]);
        auto p_begin = p.get();
#else
        Sequence<T> p(n, [](size_t){ return T{}; });
        auto p_begin = p.begin();
#endif
        auto p_end = ::std::next(p_begin, n);

#ifdef _PSTL_TEST_UNITIALIZED_DEFAULT_CONSTRUCT
        invoke_on_all_policies<>()(test_uninit_default_construct<T>(), p_begin, p_end, n,
                                   ::std::is_trivial<T>());
#endif
#ifdef _PSTL_TEST_UNITIALIZED_DEFAULT_CONSTRUCT_N
        invoke_on_all_policies<>()(test_uninit_default_construct_n<T>(), p_begin, p_end, n,
                                   ::std::is_trivial<T>());
#endif
#ifdef _PSTL_TEST_UNITIALIZED_VALUE_CONSTRUCT
        invoke_on_all_policies<>()(test_uninit_value_construct<T>(), p_begin, p_end, n,
                                   ::std::is_trivial<T>());
#endif
#ifdef _PSTL_TEST_UNITIALIZED_VALUE_CONSTRUCT_N
        invoke_on_all_policies<>()(test_uninit_value_construct_n<T>(), p_begin, p_end, n,
                                   ::std::is_trivial<T>());
#endif
    }
}

int
main()
{

#if !TEST_DPCPP_BACKEND_PRESENT
    // for user-defined types
    test_uninit_construct_by_type<Wrapper<std::int32_t>>();
    test_uninit_construct_by_type<Wrapper<::std::vector<::std::string>>>();
#endif

    // for trivial types
    test_uninit_construct_by_type<std::int8_t>();
    test_uninit_construct_by_type<float64_t>();

    return done();
}
