// -*- C++ -*-
//===-- is_heap.pass.cpp --------------------------------------------------===//
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

// Tests for is_heap, is_heap_until

#include "support/test_config.h"

#include _PSTL_TEST_HEADER(execution)
#include _PSTL_TEST_HEADER(algorithm)

#include "support/utils.h"

#include <iostream>

#if  !defined(_PSTL_TEST_IS_HEAP) && !defined(_PSTL_TEST_IS_HEAP_UNTIL)
#define _PSTL_TEST_IS_HEAP
#define _PSTL_TEST_IS_HEAP_UNTIL
#endif

using namespace TestUtils;

struct WithCmpOp
{
    std::int32_t _first;
    std::int32_t _second;
    WithCmpOp() : _first(0), _second(0){};
    explicit WithCmpOp(std::int32_t x) : _first(x), _second(x){};
    bool
    operator<(const WithCmpOp& rhs) const
    {
        return this->_first < rhs._first;
    }
};

template <typename T>
struct test_is_heap
{
    template <typename Policy, typename Iterator>
    ::std::enable_if_t<is_base_of_iterator_category_v<::std::random_access_iterator_tag, Iterator>>
    operator()(Policy&& exec, Iterator first, Iterator last)
    {
        using namespace std;
        bool expected = is_heap(first, last);
        bool actual = is_heap(exec, first, last);
        EXPECT_TRUE(expected == actual, "wrong return value from is_heap");
    }

    // is_heap works only with random access iterators
    template <typename Policy, typename Iterator>
    ::std::enable_if_t<!is_base_of_iterator_category_v<::std::random_access_iterator_tag, Iterator>>
    operator()(Policy&& /* exec */, Iterator /* first */, Iterator /* last */)
    {
    }
};

template <typename T>
struct test_is_heap_predicate
{
    template <typename Policy, typename Iterator, typename Predicate>
    ::std::enable_if_t<is_base_of_iterator_category_v<::std::random_access_iterator_tag, Iterator>>
    operator()(Policy&& exec, Iterator first, Iterator last, Predicate pred)
    {
        using namespace std;
        bool expected = is_heap(first, last, pred);
        bool actual = is_heap(exec, first, last, pred);
        EXPECT_TRUE(expected == actual, "wrong return value from is_heap with predicate");
    }

    // is_heap works only with random access iterators
    template <typename Policy, typename Iterator, typename Predicate>
    ::std::enable_if_t<!is_base_of_iterator_category_v<::std::random_access_iterator_tag, Iterator>>
    operator()(Policy&& /* exec */, Iterator /* first */, Iterator /* last */, Predicate /* pred */)
    {
    }
};

template <typename T>
struct test_is_heap_until
{
    template <typename Policy, typename Iterator>
    ::std::enable_if_t<is_base_of_iterator_category_v<::std::random_access_iterator_tag, Iterator>>
    operator()(Policy&& exec, Iterator first, Iterator last)
    {
        using namespace std;
        Iterator expected = is_heap_until(first, last);
        Iterator actual = is_heap_until(exec, first, last);
        EXPECT_TRUE(expected == actual, "wrong return value from is_heap_until");
    }

    // is_heap, is_heap_until works only with random access iterators
    template <typename Policy, typename Iterator>
    ::std::enable_if_t<!is_base_of_iterator_category_v<::std::random_access_iterator_tag, Iterator>>
    operator()(Policy&& /* exec */, Iterator /* first */, Iterator /* last */)
    {
    }
};

template <typename T>
struct test_is_heap_until_predicate
{
    template <typename Policy, typename Iterator, typename Predicate>
    ::std::enable_if_t<is_base_of_iterator_category_v<::std::random_access_iterator_tag, Iterator>>
    operator()(Policy&& exec, Iterator first, Iterator last, Predicate pred)
    {
        using namespace std;
        const Iterator expected = is_heap_until(first, last, pred);
        const Iterator actual = is_heap_until(exec, first, last, pred);
        EXPECT_TRUE(expected == actual, "wrong return value from is_heap_until with predicate");
    }

    // is_heap, is_heap_until works only with random access iterators
    template <typename Policy, typename Iterator, typename Predicate>
    ::std::enable_if_t<!is_base_of_iterator_category_v<::std::random_access_iterator_tag, Iterator>>
    operator()(Policy&& /* exec */, Iterator /* first */, Iterator /* last */, Predicate /* pred */)
    {
    }
};

template <typename T, typename Comp>
void
test_is_heap_by_type(Comp comp)
{
    using namespace std;

    const size_t max_size = 100000;
    for (size_t n = 0; n <= max_size; n = n <= 16 ? n + 1 : size_t(3.1415 * n))
    {
        Sequence<T> in(n, [](size_t v) -> T { return T(v); });

#ifdef _PSTL_TEST_IS_HEAP
        invoke_on_all_policies<0>()(test_is_heap<T>(), in.begin(), in.end());
        invoke_on_all_policies<1>()(test_is_heap_predicate<T>(), in.begin(), in.end(), comp);

        ::std::make_heap(in.begin(), in.begin() + n / 4, comp);
        invoke_on_all_policies<2>()(test_is_heap<T>(), in.cbegin(), in.cend());
        invoke_on_all_policies<3>()(test_is_heap_predicate<T>(), in.cbegin(), in.cend(), comp);

        ::std::make_heap(in.begin(), in.begin() + n / 3, comp);
        invoke_on_all_policies<4>()(test_is_heap<T>(), in.begin(), in.end());
        invoke_on_all_policies<5>()(test_is_heap_predicate<T>(), in.begin(), in.end(), comp);

        ::std::make_heap(in.begin(), in.end(), comp);
        invoke_on_all_policies<6>()(test_is_heap<T>(), in.cbegin(), in.cend());
        invoke_on_all_policies<7>()(test_is_heap_predicate<T>(), in.cbegin(), in.cend(), comp);

#endif

#ifdef _PSTL_TEST_IS_HEAP_UNTIL
        invoke_on_all_policies<8>()(test_is_heap_until<T>(), in.begin(), in.end());
        invoke_on_all_policies<9>()(test_is_heap_until_predicate<T>(), in.begin(), in.end(), comp);

        ::std::make_heap(in.begin(), in.begin() + n / 4, comp);
        invoke_on_all_policies<10>()(test_is_heap_until<T>(), in.cbegin(), in.cend());
        invoke_on_all_policies<11>()(test_is_heap_until_predicate<T>(), in.cbegin(), in.cend(), comp);

        ::std::make_heap(in.begin(), in.begin() + n / 3, comp);
        invoke_on_all_policies<12>()(test_is_heap_until<T>(), in.begin(), in.end());
        invoke_on_all_policies<13>()(test_is_heap_until_predicate<T>(), in.begin(), in.end(), comp);

        ::std::make_heap(in.begin(), in.end(), comp);
        invoke_on_all_policies<14>()(test_is_heap_until<T>(), in.cbegin(), in.cend());
        invoke_on_all_policies<15>()(test_is_heap_until_predicate<T>(), in.cbegin(), in.cend(), comp);
#endif
    }

    Sequence<T> in(max_size / 10, [](size_t) -> T { return T(1); });
#ifdef _PSTL_TEST_IS_HEAP
    invoke_on_all_policies<16>()(test_is_heap<T>(), in.begin(), in.end());
    invoke_on_all_policies<17>()(test_is_heap_predicate<T>(), in.begin(), in.end(), comp);
#endif
#ifdef _PSTL_TEST_IS_HEAP_UNTIL
    invoke_on_all_policies<18>()(test_is_heap_until<T>(), in.begin(), in.end());
    invoke_on_all_policies<19>()(test_is_heap_until_predicate<T>(), in.begin(), in.end(), comp);
#endif
}

template <typename T>
struct test_non_const_is_heap
{
    template <typename Policy, typename Iterator>
    void
    operator()(Policy&& exec, Iterator iter)
    {
        is_heap(exec, iter, iter, non_const(::std::less<T>()));
    }
};

template <typename T>
struct test_non_const_is_heap_until
{
    template <typename Policy, typename Iterator>
    void
    operator()(Policy&& exec, Iterator iter)
    {
        is_heap_until(exec, iter, iter, non_const(::std::less<T>()));
    }
};

int
main()
{
    test_is_heap_by_type<float32_t>(::std::greater<float32_t>());
    test_is_heap_by_type<WithCmpOp>(::std::less<WithCmpOp>());
    test_is_heap_by_type<std::uint64_t>([](std::uint64_t x, std::uint64_t y) { return x % 100 < y % 100; });

#ifdef _PSTL_TEST_IS_HEAP
    test_algo_basic_single<std::int32_t>(run_for_rnd<test_non_const_is_heap<std::int32_t>>());
#endif
#ifdef _PSTL_TEST_IS_HEAP_UNTIL
    test_algo_basic_single<std::int32_t>(run_for_rnd<test_non_const_is_heap_until<std::int32_t>>());
#endif

    return done();
}
