// -*- C++ -*-
//===----------------------------------------------------------------------===//
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

#include "sycl_iterator_test.h"

#if TEST_DPCPP_BACKEND_PRESENT

DEFINE_TEST(test_adjacent_find)
{
    DEFINE_TEST_CONSTRUCTOR(test_adjacent_find, 2.0f, 0.65f)

    template <typename Policy, typename Iterator, typename Size>
    void
    operator()(Policy&& exec, Iterator first, Iterator last, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        using ValueType = typename ::std::iterator_traits<Iterator>::value_type;

        auto comp = ::std::equal_to<ValueType>{};

        ValueType fill_value{0};
        ::std::for_each(host_keys.get(), host_keys.get() + n,
                        [&fill_value](ValueType& value) { value = fill_value++ % 10; });
        host_keys.update_data();

        // check with no adjacent equal elements
        Iterator result = ::std::adjacent_find(make_new_policy<new_kernel_name<Policy, 0>>(exec), first, last, comp);
        Iterator expected = last;
        wait_and_throw(exec);

        EXPECT_TRUE(result == expected, "wrong effect from adjacent_find (Test #1 no elements)");
#if _ONEDPL_DEBUG_SYCL
        ::std::cout << "got: [" << ::std::distance(first, result) << "], "
                    << "expected: [" << ::std::distance(first, expected) << "]" << ::std::endl;
 #endif // _ONEDPL_DEBUG_SYCL

        // check with the last adjacent element
        ::std::size_t max_dis = n;
        if (max_dis > 1)
        {
            *(host_keys.get() + n - 1) = *(host_keys.get() + n - 2);
            host_keys.update_data();
        }
        result = ::std::adjacent_find(make_new_policy<new_kernel_name<Policy, 0>>(exec), first, last, comp);
        expected = max_dis > 1 ? last - 2 : last;
        wait_and_throw(exec);

        EXPECT_TRUE(result == expected, "wrong effect from adjacent_find (Test #2 the last element)");
#if _ONEDPL_DEBUG_SYCL
        ::std::cout << "got: [" << ::std::distance(first, result) << "], "
                  << "expected: [" << ::std::distance(first, expected) << "]" << ::std::endl;
#endif // _ONEDPL_DEBUG_SYCL

        // check with an adjacent element
        max_dis = n;
        Iterator it{last};
        if (max_dis > 1)
        {
            it = Iterator{first + /*max_idx*/ max_dis / 2};
            *(host_keys.get() + max_dis / 2) = *(host_keys.get() + max_dis / 2 - 1);
            host_keys.update_data();
        }
        result = ::std::adjacent_find(make_new_policy<new_kernel_name<Policy, 0>>(exec), first, last, comp);
        expected = max_dis > 1 ? it - 1 : last;
        wait_and_throw(exec);

        EXPECT_TRUE(result == expected, "wrong effect from adjacent_find (Test #3 middle element)");
#if _ONEDPL_DEBUG_SYCL
        ::std::cout << "got: [" << ::std::distance(first, result) << "], "
                  << "expected: [" << ::std::distance(first, expected) << "]" << ::std::endl;
#endif // _ONEDPL_DEBUG_SYCL

        // check with an adjacent element (no predicate)
        result = ::std::adjacent_find(make_new_policy<new_kernel_name<Policy, 0>>(exec), first, last);
        wait_and_throw(exec);

        EXPECT_TRUE(result == expected, "wrong effect from adjacent_find (Test #4 middle element (no predicate))");
#if _ONEDPL_DEBUG_SYCL
        ::std::cout << "got: [" << ::std::distance(first, result) << "], "
                  << "expected: [" << ::std::distance(first, expected) << "]" << ::std::endl;
#endif // _ONEDPL_DEBUG_SYCL

        // check with the first adjacent element
        max_dis = n;
        if (max_dis > 1)
        {
            *(host_keys.get() + 1) = *host_keys.get();
            host_keys.update_data();
        }
        result = ::std::adjacent_find(make_new_policy<new_kernel_name<Policy, 0>>(exec), first, last, comp);
        expected = max_dis > 1 ? first : last;
        wait_and_throw(exec);

        EXPECT_TRUE(result == expected, "wrong effect from adjacent_find (Test #5 the first element)");
#if _ONEDPL_DEBUG_SYCL
        ::std::cout << "got: [" << ::std::distance(first, result) << "], "
                  << "expected: [" << ::std::distance(first, expected) << "]" << ::std::endl;
#endif // _ONEDPL_DEBUG_SYCL
    }
};

DEFINE_TEST(test_is_sorted_until)
{
    DEFINE_TEST_CONSTRUCTOR(test_is_sorted_until, 2.0f, 0.65f)

    template <typename Policy, typename Iterator, typename Size>
    void
    operator()(Policy&& exec, Iterator first, Iterator last, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        using ValueType = typename ::std::iterator_traits<Iterator>::value_type;

        auto comp = ::std::less<ValueType>{};

        ValueType fill_value{0};
        ::std::for_each(host_keys.get(), host_keys.get() + n,
                        [&fill_value](ValueType& value) { value = ++fill_value; });
        host_keys.update_data();

        // check sorted
        Iterator result = ::std::is_sorted_until(make_new_policy<new_kernel_name<Policy, 0>>(exec), first, last, comp);
        Iterator expected = last;
        wait_and_throw(exec);

        EXPECT_TRUE(result == expected, "wrong effect from is_sorted_until (Test #1 sorted sequence)");
#if _ONEDPL_DEBUG_SYCL
        ::std::cout << "got: [" << ::std::distance(first, result) << "], "
                  << "expected: [" << ::std::distance(first, expected) << "]" << ::std::endl;
#endif // _ONEDPL_DEBUG_SYCL

        // check unsorted: the last element
        ::std::size_t max_dis = n;
        if (max_dis > 1)
        {
            *(host_keys.get() + n - 1) = ValueType{0};
            host_keys.update_data();
        }
        result = ::std::is_sorted_until(make_new_policy<new_kernel_name<Policy, 1>>(exec), first, last, comp);
        expected = max_dis > 1 ? last - 1 : last;
        wait_and_throw(exec);

        EXPECT_TRUE(result == expected,
                    "wrong effect from is_sorted_until (Test #2 unsorted sequence - the last element)");
#if _ONEDPL_DEBUG_SYCL
        ::std::cout << "got: [" << ::std::distance(first, result) << "], "
                  << "expected: [" << ::std::distance(first, expected) << "]" << ::std::endl;
#endif // _ONEDPL_DEBUG_SYCL

        // check unsorted: the middle element
        max_dis = n;
        Iterator it{last};
        if (max_dis > 1)
        {
            it = Iterator{first + /*max_idx*/ max_dis / 2};
            *(host_keys.get() + /*max_idx*/ max_dis / 2) = ValueType{0};
            host_keys.update_data();
        }
        result = ::std::is_sorted_until(make_new_policy<new_kernel_name<Policy, 2>>(exec), first, last, comp);
        expected = it;
        wait_and_throw(exec);

        EXPECT_TRUE(result == expected,
                    "wrong effect from is_sorted_until (Test #3 unsorted sequence - the middle element)");
#if _ONEDPL_DEBUG_SYCL
        ::std::cout << "got: [" << ::std::distance(first, result) << "], "
                  << "expected: [" << ::std::distance(first, expected) << "]" << ::std::endl;
#endif // _ONEDPL_DEBUG_SYCL

        // check unsorted: the middle element (no predicate)
        result = ::std::is_sorted_until(make_new_policy<new_kernel_name<Policy, 3>>(exec), first, last);
        wait_and_throw(exec);

        EXPECT_TRUE(
            result == expected,
            "wrong effect from is_sorted_until (Test #4 unsorted sequence - the middle element (no predicate))");
#if _ONEDPL_DEBUG_SYCL
        ::std::cout << "got: [" << ::std::distance(first, result) << "], "
                  << "expected: [" << ::std::distance(first, expected) << "]" << ::std::endl;
#endif // _ONEDPL_DEBUG_SYCL

        // check unsorted: the first element
        if (n > 1)
        {
            *(host_keys.get() + 1) = ValueType{0};
            host_keys.update_data();
        }
        result = ::std::is_sorted_until(make_new_policy<new_kernel_name<Policy, 4>>(exec), first, last, comp);
        expected = n > 1 ? first + 1 : last;
        wait_and_throw(exec);

        EXPECT_TRUE(result == expected,
                    "wrong effect from is_sorted_until (Test #5 unsorted sequence - the first element)");
#if _ONEDPL_DEBUG_SYCL
        ::std::cout << "got: [" << ::std::distance(first, result) << "], "
                  << "expected: [" << ::std::distance(first, expected) << "]" << ::std::endl;
#endif // _ONEDPL_DEBUG_SYCL
    }
};

DEFINE_TEST(test_is_sorted)
{
    DEFINE_TEST_CONSTRUCTOR(test_is_sorted, 2.0f, 0.65f)

    template <typename Policy, typename Iterator, typename Size>
    void
    operator()(Policy&& exec, Iterator first, Iterator last, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        using ValueType = typename ::std::iterator_traits<Iterator>::value_type;

        auto comp = ::std::less<ValueType>{};

        ValueType fill_value{ 0 };
        ::std::for_each(host_keys.get(), host_keys.get() + n,
                        [&fill_value](ValueType& value) { value = ++fill_value; });
        host_keys.update_data();

        // check sorted
        bool result_bool = ::std::is_sorted(make_new_policy<new_kernel_name<Policy, 0>>(exec), first, last, comp);
        bool expected_bool = true;
        wait_and_throw(exec);

        EXPECT_TRUE(result_bool == expected_bool, "wrong effect from is_sorted (Test #1 sorted sequence)");
#if _ONEDPL_DEBUG_SYCL
        ::std::cout << "got: " << result_bool << ", "
                    << "expected: " << expected_bool << ::std::endl;
#endif // _ONEDPL_DEBUG_SYCL

        // check unsorted: the last element
        ::std::size_t max_dis = n;
        if (max_dis > 1)
        {
            *(host_keys.get() + n - 1) = ValueType{0};
            host_keys.update_data();
        }
        result_bool = ::std::is_sorted(make_new_policy<new_kernel_name<Policy, 1>>(exec), first, last, comp);
        expected_bool = max_dis > 1 ? false : true;
        wait_and_throw(exec);

        EXPECT_TRUE(result_bool == expected_bool,
                    "wrong effect from is_sorted (Test #2 unsorted sequence - the last element)");
#if _ONEDPL_DEBUG_SYCL
        ::std::cout << "got: " << result_bool << ", "
                    << "expected: " << expected_bool << ::std::endl;
#endif // _ONEDPL_DEBUG_SYCL

        // check unsorted: the middle element
        max_dis = n;
        if (max_dis > 1)
        {
            *(host_keys.get() + max_dis / 2) = ValueType{0};
            host_keys.update_data();
        }
        result_bool = ::std::is_sorted(make_new_policy<new_kernel_name<Policy, 2>>(exec), first, last, comp);
        expected_bool = max_dis > 1 ? false : true;
        wait_and_throw(exec);

        EXPECT_TRUE(result_bool == expected_bool,
                    "wrong effect from is_sorted (Test #3 unsorted sequence - the middle element)");
#if _ONEDPL_DEBUG_SYCL
        ::std::cout << "got: " << result_bool << ", "
                    << "expected: " << expected_bool << ::std::endl;
#endif // _ONEDPL_DEBUG_SYCL

        // check unsorted: the middle element (no predicate)
        result_bool = ::std::is_sorted(make_new_policy<new_kernel_name<Policy, 3>>(exec), first, last);
        wait_and_throw(exec);

        EXPECT_TRUE(result_bool == expected_bool,
                    "wrong effect from is_sorted (Test #4 unsorted sequence - the middle element (no predicate))");
#if _ONEDPL_DEBUG_SYCL
        ::std::cout << "got: " << result_bool << ", "
                    << "expected: " << expected_bool << ::std::endl;
#endif // _ONEDPL_DEBUG_SYCL

        // check unsorted: the first element
        max_dis = n;
        if (max_dis > 1)
        {
            *(host_keys.get() + 1) = ValueType{0};
            host_keys.update_data();
        }
        result_bool = ::std::is_sorted(make_new_policy<new_kernel_name<Policy, 4>>(exec), first, last, comp);
        expected_bool = max_dis > 1 ? false : true;
        wait_and_throw(exec);

        EXPECT_TRUE(result_bool == expected_bool,
                    "wrong effect from is_sorted Test #5 unsorted sequence - the first element");
#if _ONEDPL_DEBUG_SYCL
        ::std::cout << "got: " << result_bool << ", "
                    << "expected: " << expected_bool << ::std::endl;
#endif // _ONEDPL_DEBUG_SYCL
    }
};

template <typename T1, typename Size>
struct IsMinusOneEqTo
{
    Size n;
    bool operator()(T1 x) const
    {
        return x == n - 1;
    }
};

template <typename T1>
struct IsEqMinusOne
{
    bool operator()(T1 x) const
    {
        return x == -1;
    }
};

template <typename T1>
struct MultipleOf2
{
    bool operator()(T1 x) const
    {
        return x % 2 == 0;
    }
};

DEFINE_TEST(test_any_all_none_of)
{
    DEFINE_TEST_CONSTRUCTOR(test_any_all_none_of, 2.0f, 0.65f)

    template <typename Policy, typename Iterator1, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;

        ::std::iota(host_keys.get(), host_keys.get() + n, T1(0));
        host_keys.update_data();

        // empty sequence case
        if (n == 1)
        {
            auto res0 = std::any_of(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, first1, IsMinusOneEqTo<T1, Size>{n});
            wait_and_throw(exec);

            EXPECT_TRUE(!res0, "wrong effect from any_of_0");
            res0 = std::none_of(make_new_policy<new_kernel_name<Policy, 1>>(exec), first1, first1, IsEqMinusOne<T1>{});
            wait_and_throw(exec);

            EXPECT_TRUE(res0, "wrong effect from none_of_0");
            res0 = std::all_of(make_new_policy<new_kernel_name<Policy, 2>>(exec), first1, first1, MultipleOf2<T1>{});
            wait_and_throw(exec);

            EXPECT_TRUE(res0, "wrong effect from all_of_0");
        }
        // any_of
        auto res1 = std::any_of(make_new_policy<new_kernel_name<Policy, 3>>(exec), first1, last1, IsMinusOneEqTo<T1, Size>{n});
        wait_and_throw(exec);

        EXPECT_TRUE(res1, "wrong effect from any_of_1");
        auto res2 = std::any_of(make_new_policy<new_kernel_name<Policy, 4>>(exec), first1, last1, IsEqMinusOne<T1>{});
        wait_and_throw(exec);

        EXPECT_TRUE(!res2, "wrong effect from any_of_2");
        auto res3 = std::any_of(make_new_policy<new_kernel_name<Policy, 5>>(exec), first1, last1, MultipleOf2<T1>{});
        wait_and_throw(exec);

        EXPECT_TRUE(res3, "wrong effect from any_of_3");

        //none_of
        auto res4 = std::none_of(make_new_policy<new_kernel_name<Policy, 6>>(exec), first1, last1, IsEqMinusOne<T1>{});
        wait_and_throw(exec);

        EXPECT_TRUE(res4, "wrong effect from none_of");

        //all_of
        auto res5 = std::all_of(make_new_policy<new_kernel_name<Policy, 7>>(exec), first1, last1, MultipleOf2<T1>{});
        wait_and_throw(exec);

        EXPECT_TRUE(n == 1 || !res5, "wrong effect from all_of");
    }
};

DEFINE_TEST(test_find_if)
{
    DEFINE_TEST_CONSTRUCTOR(test_find_if, 2.0f, 0.65f)

    template <typename Policy, typename Iterator1, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;

        ::std::iota(host_keys.get(), host_keys.get() + n, T1(0));
        host_keys.update_data();

        // empty sequence case
        if (n == 1)
        {
            auto res0 = std::find_if(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, first1, IsMinusOneEqTo<T1, Size>{n});
            wait_and_throw(exec);

            EXPECT_TRUE(res0 == first1, "wrong effect from find_if_0");
            res0 = ::std::find(make_new_policy<new_kernel_name<Policy, 1>>(exec), first1, first1, T1(1));
            wait_and_throw(exec);

            EXPECT_TRUE(res0 == first1, "wrong effect from find_0");
        }
        // find_if
        auto res1 = std::find_if(make_new_policy<new_kernel_name<Policy, 2>>(exec), first1, last1, IsMinusOneEqTo<T1, Size>{n});
        wait_and_throw(exec);

        EXPECT_TRUE((res1 - first1) == n - 1, "wrong effect from find_if_1");

        auto res2 = std::find_if(make_new_policy<new_kernel_name<Policy, 3>>(exec), first1, last1, IsEqMinusOne<T1>{});
        wait_and_throw(exec);

        EXPECT_TRUE(res2 == last1, "wrong effect from find_if_2");

        auto res3 = std::find_if(make_new_policy<new_kernel_name<Policy, 4>>(exec), first1, last1, MultipleOf2<T1>{});
        wait_and_throw(exec);

        EXPECT_TRUE(res3 == first1, "wrong effect from find_if_3");

        //find
        auto res4 = ::std::find(make_new_policy<new_kernel_name<Policy, 5>>(exec), first1, last1, T1(-1));
        wait_and_throw(exec);

        EXPECT_TRUE(res4 == last1, "wrong effect from find");
    }
};

DEFINE_TEST(test_search_n)
{
    DEFINE_TEST_CONSTRUCTOR(test_search_n, 2.0f, 0.65f)

    template <typename Policy, typename Iterator, typename Size>
    void
    operator()(Policy&& exec, Iterator first, Iterator last, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        typedef typename ::std::iterator_traits<Iterator>::value_type T;

        ::std::iota(host_keys.get(), host_keys.get() + n, T(100));

        // Search for sequence at the end
        {
            auto start = (n > 3) ? (n / 3 * 2) : (n - 1);

            ::std::fill(host_keys.get() + start, host_keys.get() + n, T(11));
            host_keys.update_data();

            auto res = ::std::search_n(make_new_policy<new_kernel_name<Policy, 0>>(exec), first, last, n - start, T(11));
            wait_and_throw(exec);

            EXPECT_TRUE(res - first == start, "wrong effect from search_1");
        }
        // Search for sequence in the middle
        {
            auto start = (n > 3) ? (n / 3) : (n - 1);
            auto end = (n > 3) ? (n / 3 * 2) : n;

            ::std::fill(host_keys.get() + start, host_keys.get() + end, T(22));
            host_keys.update_data();

            auto res = ::std::search_n(make_new_policy<new_kernel_name<Policy, 1>>(exec), first, last, end - start, T(22));
            wait_and_throw(exec);

            EXPECT_TRUE(res - first == start, "wrong effect from search_20");

            // Search for sequence of lesser size
            res = ::std::search_n(make_new_policy<new_kernel_name<Policy, 2>>(exec), first, last,
                                ::std::max(end - start - 1, (size_t)1), T(22));
            wait_and_throw(exec);

            EXPECT_TRUE(res - first == start, "wrong effect from search_21");
        }
        // Search for sequence at the beginning
        {
            auto end = n / 3;

            ::std::fill(host_keys.get(), host_keys.get() + end, T(33));
            host_keys.update_data();

            auto res = ::std::search_n(make_new_policy<new_kernel_name<Policy, 3>>(exec), first, last, end, T(33));
            wait_and_throw(exec);

            EXPECT_TRUE(res == first, "wrong effect from search_3");
        }
        // Search for sequence that covers the whole range
        {
            ::std::fill(host_keys.get(), host_keys.get() + n, T(44));
            host_keys.update_data();

            auto res = ::std::search_n(make_new_policy<new_kernel_name<Policy, 4>>(exec), first, last, n, T(44));
            wait_and_throw(exec);

            EXPECT_TRUE(res == first, "wrong effect from search_4");
        }
        // Search for sequence which is not there
        {
            auto res = ::std::search_n(make_new_policy<new_kernel_name<Policy, 5>>(exec), first, last, 2, T(55));
            wait_and_throw(exec);

            EXPECT_TRUE(res == last, "wrong effect from search_50");

            // Sequence is there but of lesser size(see search_n_3)
            res = ::std::search_n(make_new_policy<new_kernel_name<Policy, 6>>(exec), first, last, (n / 3 + 1), T(33));
            wait_and_throw(exec);

            EXPECT_TRUE(res == last, "wrong effect from search_51");
        }

        // empty sequence case
        {
            auto res = ::std::search_n(make_new_policy<new_kernel_name<Policy, 7>>(exec), first, first, 1, T(5 + n - 1));
            wait_and_throw(exec);

            EXPECT_TRUE(res == first, "wrong effect from search_6");
        }
        // 2 distinct sequences, must find the first one
        if (n > 10)
        {
            auto start1 = n / 6;
            auto end1 = n / 3;

            auto start2 = (2 * n) / 3;
            auto end2 = (5 * n) / 6;

            ::std::fill(host_keys.get() + start1, host_keys.get() + end1, T(66));
            ::std::fill(host_keys.get() + start2, host_keys.get() + end2, T(66));
            host_keys.update_data();

            auto res = ::std::search_n(make_new_policy<new_kernel_name<Policy, 8>>(exec), first, last,
                                     ::std::min(end1 - start1, end2 - start2), T(66));
            wait_and_throw(exec);

            EXPECT_TRUE(res - first == start1, "wrong effect from search_7");
        }

        if (n == 10)
        {
            auto seq_len = 3;

            // Should fail when searching for sequence which is placed before our first iterator.
            ::std::fill(host_keys.get(), host_keys.get() + seq_len, T(77));
            host_keys.update_data();

            auto res = ::std::search_n(make_new_policy<new_kernel_name<Policy, 9>>(exec), first + 1, last, seq_len, T(77));
            wait_and_throw(exec);

            EXPECT_TRUE(res == last, "wrong effect from search_8");
        }
    }
};

DEFINE_TEST(test_is_heap)
{
    DEFINE_TEST_CONSTRUCTOR(test_is_heap, 2.0f, 0.65f)

    template <typename Policy, typename Iterator, typename Size>
    void
    operator()(Policy&& exec, Iterator first, Iterator last, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        using ValueType = typename ::std::iterator_traits<Iterator>::value_type;

        ::std::iota(host_keys.get(), host_keys.get() + n, ValueType(0));
        ::std::make_heap(host_keys.get(), host_keys.get());
        host_keys.update_data();

        auto actual = ::std::is_heap(make_new_policy<new_kernel_name<Policy, 0>>(exec), first, last);
        // True only when n == 1
        wait_and_throw(exec);

        EXPECT_TRUE(actual == (n == 1), "wrong result of is_heap_11");

        actual = ::std::is_heap(make_new_policy<new_kernel_name<Policy, 1>>(exec), first, first);
        wait_and_throw(exec);

        EXPECT_TRUE(actual == true, "wrong result of is_heap_12");

        if (n <= 5)
            return;

        host_keys.retrieve_data();
        ::std::make_heap(host_keys.get(), host_keys.get() + n / 2);
        host_keys.update_data();

        actual = ::std::is_heap(make_new_policy<new_kernel_name<Policy, 0>>(exec), first, last);
        wait_and_throw(exec);

        EXPECT_TRUE(actual == false, "wrong result of is_heap_21");

        auto end = first + n / 2;
        actual = ::std::is_heap(make_new_policy<new_kernel_name<Policy, 1>>(exec), first, end);
        wait_and_throw(exec);

        EXPECT_TRUE(actual == true, "wrong result of is_heap_22");

        host_keys.retrieve_data();
        ::std::make_heap(host_keys.get(), host_keys.get() + n);
        host_keys.update_data();

        actual = ::std::is_heap(make_new_policy<new_kernel_name<Policy, 2>>(exec), first, last);
        wait_and_throw(exec);

        EXPECT_TRUE(actual == true, "wrong result of is_heap_3");
    }
};

DEFINE_TEST(test_is_heap_until)
{
    DEFINE_TEST_CONSTRUCTOR(test_is_heap_until, 2.0f, 0.65f)

    template <typename Policy, typename Iterator, typename Size>
    void
    operator()(Policy&& exec, Iterator first, Iterator last, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        using ValueType = typename ::std::iterator_traits<Iterator>::value_type;

        ::std::iota(host_keys.get(), host_keys.get() + n, ValueType(0));
        ::std::make_heap(host_keys.get(), host_keys.get());
        host_keys.update_data();

        auto actual = ::std::is_heap_until(make_new_policy<new_kernel_name<Policy, 0>>(exec), first, last);
        wait_and_throw(exec);

        // first element is always a heap
        EXPECT_TRUE(actual == first + 1, "wrong result of is_heap_until_1");

        if (n <= 5)
            return;

        host_keys.retrieve_data();
        ::std::make_heap(host_keys.get(), host_keys.get() + n / 2);
        host_keys.update_data();

        actual = ::std::is_heap_until(make_new_policy<new_kernel_name<Policy, 2>>(exec), first, last);
        wait_and_throw(exec);

        EXPECT_TRUE(actual == (first + n / 2), "wrong result of is_heap_until_2");

        host_keys.retrieve_data();
        ::std::make_heap(host_keys.get(), host_keys.get() + n);
        host_keys.update_data();

        actual = ::std::is_heap_until(make_new_policy<new_kernel_name<Policy, 3>>(exec), first, last);
        wait_and_throw(exec);

        EXPECT_TRUE(actual == last, "wrong result of is_heap_until_3");
    }
};

DEFINE_TEST(test_equal)
{
    DEFINE_TEST_CONSTRUCTOR(test_equal, 2.0f, 0.65f)

    template <typename Policy, typename Iterator1, typename Iterator2, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 /* last1 */, Iterator2 first2, Iterator2 /* last2 */, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);
        TestDataTransfer<UDTKind::eVals, Size> host_vals(*this, n);

        using T = typename ::std::iterator_traits<Iterator1>::value_type;
        auto value = T(42);

        auto new_start = n / 3;
        auto new_end = n / 2;

        ::std::fill(host_keys.get(), host_keys.get() + n, value);
        ::std::fill(host_vals.get(), host_vals.get() + n, T{0});
        ::std::fill(host_vals.get() + new_start, host_vals.get() + new_end, value);
        update_data(host_keys, host_vals);

        auto expected  = new_end - new_start > 0;
        auto result = ::std::equal(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1 + new_start,
                                   first1 + new_end, first2 + new_start);
        wait_and_throw(exec);

        EXPECT_TRUE(expected == result, "wrong effect from equal with 3 iterators");
        result = ::std::equal(make_new_policy<new_kernel_name<Policy, 1>>(exec), first1 + new_start, first1 + new_end,
                              first2 + new_start, first2 + new_end);
        wait_and_throw(exec);

        EXPECT_TRUE(expected == result, "wrong effect from equal with 4 iterators");
    }
};

DEFINE_TEST(test_find_first_of)
{
    DEFINE_TEST_CONSTRUCTOR(test_find_first_of, 2.0f, 0.65f)

    template <typename Policy, typename Iterator1, typename Iterator2, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 last2, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);
        TestDataTransfer<UDTKind::eVals, Size> host_vals(*this, n);

        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;

        // Reset values after previous execution
        ::std::fill(host_keys.get(), host_keys.get() + n, T1(0));
        host_keys.update_data();

        if (n < 2)
        {
            ::std::iota(host_vals.get(), host_vals.get() + n, T1(5));
            host_vals.update_data();

            auto res =
                ::std::find_first_of(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, first1, first2, last2);
            wait_and_throw(exec);

            EXPECT_TRUE(res == first1, "Wrong effect from find_first_of_1");
        }
        else if (n >= 2 && n < 10)
        {
            auto res = ::std::find_first_of(make_new_policy<new_kernel_name<Policy, 1>>(exec), first1, last1,
                                            first2, first2);
            wait_and_throw(exec);

            EXPECT_TRUE(res == last1, "Wrong effect from find_first_of_2");

            // No matches
            ::std::iota(host_vals.get(), host_vals.get() + n, T1(5));
            host_vals.update_data();

            res = ::std::find_first_of(make_new_policy<new_kernel_name<Policy, 2>>(exec), first1, last1, first2, last2);
            wait_and_throw(exec);

            EXPECT_TRUE(res == last1, "Wrong effect from find_first_of_3");
        }
        else if (n >= 10)
        {
            ::std::iota(host_vals.get(), host_vals.get() + n, T1(5));
            host_vals.update_data();

            auto pos1 = n / 5;
            auto pos2 = 3 * n / 5;
            auto num = 3;

            ::std::iota(host_keys.get() + pos2, host_keys.get() + pos2 + num, T1(7));
            host_keys.update_data();

            auto res = ::std::find_first_of(make_new_policy<new_kernel_name<Policy, 3>>(exec), first1, last1, first2, last2);
            wait_and_throw(exec);

            EXPECT_TRUE(res == first1 + pos2, "Wrong effect from find_first_of_4");

            // Add second match
            ::std::iota(host_keys.get() + pos1, host_keys.get() + pos1 + num, T1(6));
            host_keys.update_data();

            res = ::std::find_first_of(make_new_policy<new_kernel_name<Policy, 4>>(exec), first1, last1, first2, last2);
            wait_and_throw(exec);

            EXPECT_TRUE(res == first1 + pos1, "Wrong effect from find_first_of_5");
        }
    }
};

DEFINE_TEST(test_search)
{
    DEFINE_TEST_CONSTRUCTOR(test_search, 2.0f, 0.65f)

    template <typename Policy, typename Iterator1, typename Iterator2, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 last2, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);
        TestDataTransfer<UDTKind::eVals, Size> host_vals(*this, n);

        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;

        ::std::iota(host_keys.get(), host_keys.get() + n, T1(5));
        ::std::iota(host_vals.get(), host_vals.get() + n, T1(0));
        update_data(host_keys, host_vals);

        // empty sequence case
        if (n == 1)
        {
            auto res0 = ::std::search(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, first1, first2, last2);
            wait_and_throw(exec);

            EXPECT_TRUE(res0 == first1, "wrong effect from search_00");
            res0 = ::std::search(make_new_policy<new_kernel_name<Policy, 1>>(exec), first1, last1, first2, first2);
            wait_and_throw(exec);

            EXPECT_TRUE(res0 == first1, "wrong effect from search_01");
        }
        auto res1 = ::std::search(make_new_policy<new_kernel_name<Policy, 1>>(exec), first1, last1, first2, last2);
        EXPECT_TRUE(res1 == last1, "wrong effect from search_1");
        if (n > 10)
        {
            // first n-10 elements of the subsequence are at the beginning of first sequence
            auto res2 = ::std::search(make_new_policy<new_kernel_name<Policy, 3>>(exec), first1, last1, first2 + 10, last2);
            wait_and_throw(exec);

            EXPECT_TRUE(res2 - first1 == 5, "wrong effect from search_2");
        }
        // subsequence consists of one element (last one)
        auto res3 = ::std::search(make_new_policy<new_kernel_name<Policy, 4>>(exec), first1, last1, last1 - 1, last1);
        wait_and_throw(exec);

        EXPECT_TRUE(last1 - res3 == 1, "wrong effect from search_3");

        // first sequence contains 2 almost similar parts
        if (n > 5)
        {
            ::std::iota(host_keys.get() + n / 2, host_keys.get() + n, T1(5));
            host_keys.update_data();

            auto res4 = ::std::search(make_new_policy<new_kernel_name<Policy, 5>>(exec), first1, last1, first2 + 5, first2 + 6);
            wait_and_throw(exec);

            EXPECT_TRUE(res4 == first1, "wrong effect from search_4");
        }
    }
};

DEFINE_TEST(test_mismatch)
{
    DEFINE_TEST_CONSTRUCTOR(test_mismatch, 2.0f, 0.65f)

    template <typename Policy, typename Iterator1, typename Iterator2, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 last2, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);
        TestDataTransfer<UDTKind::eVals, Size> host_vals(*this, n);

        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;

        ::std::iota(host_keys.get(), host_keys.get() + n, T1(5));
        ::std::iota(host_vals.get(), host_vals.get() + n, T1(0));
        update_data(host_keys, host_vals);

        // empty sequence case
        if (n == 1)
        {
            auto res0 = ::std::mismatch(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, first1, first2, last2);
            wait_and_throw(exec);

            EXPECT_TRUE(res0.first == first1 && res0.second == first2, "wrong effect from mismatch_00");
            res0 = ::std::mismatch(make_new_policy<new_kernel_name<Policy, 1>>(exec), first1, last1, first2, first2);
            wait_and_throw(exec);

            EXPECT_TRUE(res0.first == first1 && res0.second == first2, "wrong effect from mismatch_01");
        }
        auto res1 = ::std::mismatch(make_new_policy<new_kernel_name<Policy, 2>>(exec), first1, last1, first2, last2);
        EXPECT_TRUE(res1.first == first1 && res1.second == first2, "wrong effect from mismatch_1");
        if (n > 5)
        {
            // first n-10 elements of the subsequence are at the beginning of first sequence
            auto res2 = ::std::mismatch(make_new_policy<new_kernel_name<Policy, 3>>(exec), first1, last1, first2 + 5, last2);
            wait_and_throw(exec);

            EXPECT_TRUE(res2.first == last1 - 5 && res2.second == last2, "wrong effect from mismatch_2");
        }
    }
};

DEFINE_TEST(test_find_end)
{
    DEFINE_TEST_CONSTRUCTOR(test_find_end, 2.0f, 0.65f)

    template <typename Policy, typename Iterator1, typename Iterator2, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 last2, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);
        TestDataTransfer<UDTKind::eVals, Size> host_vals(*this, n);

        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;
        typedef typename ::std::iterator_traits<Iterator2>::value_type T2;

        // Reset after previous run
        {
            ::std::fill(host_keys.get(), host_keys.get() + n, T1(0));
        }

        if (n <= 2)
        {
            ::std::iota(host_vals.get(), host_vals.get() + n, T2(10));
            host_vals.update_data();

            // Empty subsequence
            auto res = ::std::find_end(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, last1, first2, first2);
            wait_and_throw(exec);

            EXPECT_TRUE(res == last1, "Wrong effect from find_end_1");

            return;
        }

        if (n > 2 && n < 10)
        {
            // re-write the sequence after previous run
            ::std::iota(host_keys.get(), host_keys.get() + n, T1(0));
            ::std::iota(host_vals.get(), host_vals.get() + n, T2(10));
            update_data(host_keys, host_vals);

            // No subsequence
            auto res = ::std::find_end(make_new_policy<new_kernel_name<Policy, 1>>(exec), first1, last1, first2, first2 + n / 2);
            wait_and_throw(exec);

            EXPECT_TRUE(res == last1, "Wrong effect from find_end_2");

            // Whole sequence is matched
            ::std::iota(host_keys.get(), host_keys.get() + n, T1(10));
            host_keys.update_data();

            res = ::std::find_end(make_new_policy<new_kernel_name<Policy, 2>>(exec), first1, last1, first2, last2);
            wait_and_throw(exec);

            EXPECT_TRUE(res == first1, "Wrong effect from find_end_3");

            return;
        }

        if (n >= 10)
        {
            ::std::iota(host_vals.get(), host_vals.get() + n / 5, T2(20));
            host_vals.update_data();

            // Match at the beginning
            {
                auto start = host_keys.get();
                auto end = host_keys.get() + n / 5;
                ::std::iota(start, end, T1(20));
                host_keys.update_data();

                auto res = ::std::find_end(make_new_policy<new_kernel_name<Policy, 3>>(exec), first1, last1, first2,
                                           first2 + n / 5);
                wait_and_throw(exec);

                EXPECT_TRUE(res == first1, "Wrong effect from find_end_4");
            }

            // 2 matches: at the beginning and in the middle, should return the latter
            {
                auto start = host_keys.get() + 2 * n / 5;
                auto end = host_keys.get() + 3 * n / 5;
                ::std::iota(start, end, T1(20));
                host_keys.update_data();


                auto res = ::std::find_end(make_new_policy<new_kernel_name<Policy, 4>>(exec), first1, last1, first2,
                                           first2 + n / 5);
                wait_and_throw(exec);

                EXPECT_TRUE(res == first1 + 2 * n / 5, "Wrong effect from find_end_5");
            }

            // 3 matches: at the beginning, in the middle and at the end, should return the latter
            {
                auto start = host_keys.get() + 4 * n / 5;
                auto end = host_keys.get() + n;
                ::std::iota(start, end, T1(20));
                host_keys.update_data();

                auto res = ::std::find_end(make_new_policy<new_kernel_name<Policy, 5>>(exec), first1, last1, first2,
                                           first2 + n / 5);
                wait_and_throw(exec);

                EXPECT_TRUE(res == first1 + 4 * n / 5, "Wrong effect from find_end_6");
            }
        }
    }
};

#endif // TEST_DPCPP_BACKEND_PRESENT

#if TEST_DPCPP_BACKEND_PRESENT
template <sycl::usm::alloc alloc_type>
void
test_usm_and_buffer()
{
    using ValueType = ::std::int32_t;

    // test1buffer
    PRINT_DEBUG("test_any_all_none_of");
    test1buffer<alloc_type, test_any_all_none_of<ValueType>>();
    PRINT_DEBUG("test_is_sorted");
    test1buffer<alloc_type, test_is_sorted<ValueType>>();
    PRINT_DEBUG("test_is_heap");
    test1buffer<alloc_type, test_is_heap<ValueType>>();
    PRINT_DEBUG("test_find_if");
    test1buffer<alloc_type, test_find_if<ValueType>>();
    PRINT_DEBUG("test_adjacent_find");
    test1buffer<alloc_type, test_adjacent_find<ValueType>>();
    PRINT_DEBUG("test_is_sorted_until");
    test1buffer<alloc_type, test_is_sorted_until<ValueType>>();
    PRINT_DEBUG("test_search_n");
    test1buffer<alloc_type, test_search_n<ValueType>>();
    PRINT_DEBUG("test_is_heap_until");
    test1buffer<alloc_type, test_is_heap_until<ValueType>>();
    print_debug("test_is_heap");
    test1buffer<alloc_type, test_is_heap<ValueType>>();

    //test2buffers
    PRINT_DEBUG("test_equal");
    test2buffers<alloc_type, test_equal<ValueType>>();
    PRINT_DEBUG("test_mismatch");
    test2buffers<alloc_type, test_mismatch<ValueType>>();
    PRINT_DEBUG("test_search");
    test2buffers<alloc_type, test_search<ValueType>>();
    PRINT_DEBUG("test_find_end");
    test2buffers<alloc_type, test_find_end<ValueType>>();
    PRINT_DEBUG("test_find_first_of");
    test2buffers<alloc_type, test_find_first_of<ValueType>>();
}
#endif // TEST_DPCPP_BACKEND_PRESENT

std::int32_t
main()
{
    try
    {
#if TEST_DPCPP_BACKEND_PRESENT
        //TODO: There is the over-testing here - each algorithm is run with sycl::buffer as well.
        //So, in case of a couple of 'test_usm_and_buffer' call we get double-testing case with sycl::buffer.

        // Run tests for USM shared memory
        test_usm_and_buffer<sycl::usm::alloc::shared>();
        // Run tests for USM device memory
        test_usm_and_buffer<sycl::usm::alloc::device>();
#endif // TEST_DPCPP_BACKEND_PRESENT
    }
    catch (const ::std::exception& exc)
    {
        std::cout << "Exception: " << exc.what() << std::endl;
        return EXIT_FAILURE;
    }

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
