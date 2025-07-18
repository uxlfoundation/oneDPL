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

DEFINE_TEST(test_reduce)
{
    DEFINE_TEST_CONSTRUCTOR(test_reduce, 2.0f, 0.80f)

    template <typename Policy, typename Iterator1, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 /* last1 */, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;
        auto value = T1(2);

        ::std::fill(host_keys.get(), host_keys.get() + n, T1(0));
        ::std::fill(host_keys.get() + (n / 3), host_keys.get() + (n / 2), value);
        host_keys.update_data();

        // without initial value
        auto result1 = std::reduce(CLONE_TEST_POLICY_IDX(exec, 0), first1 + (n / 3), first1 + (n / 2));
        wait_and_throw(exec);

        EXPECT_TRUE(result1 == value * (n / 2 - n / 3), "wrong effect from reduce (1)");

        // with initial value
        auto init = T1(42);
        auto result2 = std::reduce(CLONE_TEST_POLICY_IDX(exec, 0), first1 + (n / 3), first1 + (n / 2), init);
        wait_and_throw(exec);

        EXPECT_TRUE(result2 == init + value * (n / 2 - n / 3), "wrong effect from reduce (2)");
    }
};

DEFINE_TEST(test_transform_reduce_unary)
{
    DEFINE_TEST_CONSTRUCTOR(test_transform_reduce_unary, 2.0f, 0.80f)

    template <typename Policy, typename Iterator1, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;
        auto value = T1(1);

        ::std::fill(host_keys.get(), host_keys.get() + n, value);
        host_keys.update_data();

        auto result = std::transform_reduce(CLONE_TEST_POLICY_IDX(exec, 0), first1, last1, T1(42),
                                            Plus(), ::std::negate<T1>());
        wait_and_throw(exec);

        EXPECT_TRUE(result == 42 - n, "wrong effect from transform_reduce (unary + binary)");
    }
};

DEFINE_TEST(test_min_element)
{
    DEFINE_TEST_CONSTRUCTOR(test_min_element, 2.0f, 0.80f)

    template <typename Policy, typename Iterator, typename Size>
    void
    operator()(Policy&& exec, Iterator first, Iterator last, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        using IteratorValueType = typename ::std::iterator_traits<Iterator>::value_type;

        IteratorValueType fill_value = IteratorValueType{ static_cast<IteratorValueType>(::std::distance(first, last)) };

        ::std::for_each(host_keys.get(), host_keys.get() + n,
            [&fill_value](IteratorValueType& it) { it = fill_value-- % 10 + 1; });

        ::std::size_t min_dis = n;
        if (min_dis)
        {
            *(host_keys.get() + min_dis / 2) = IteratorValueType{/*min_val*/ 0 };
            *(host_keys.get() + n - 1) = IteratorValueType{/*2nd min*/ 0 };
        }
        host_keys.update_data();

        auto result_min = std::min_element(CLONE_TEST_POLICY_IDX(exec, 0), first, last);
        wait_and_throw(exec);

        host_keys.retrieve_data();

        auto expected_min = ::std::min_element(host_keys.get(), host_keys.get() + n);

        EXPECT_EQ(expected_min - host_keys.get(), result_min - first, "wrong effect from min_element");
    }
};

DEFINE_TEST(test_max_element)
{
    DEFINE_TEST_CONSTRUCTOR(test_max_element, 2.0f, 0.80f)

    template <typename Policy, typename Iterator, typename Size>
    void
    operator()(Policy&& exec, Iterator first, Iterator last, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        using IteratorValueType = typename ::std::iterator_traits<Iterator>::value_type;

        IteratorValueType fill_value = IteratorValueType{0};
        ::std::for_each(host_keys.get(), host_keys.get() + n,
                        [&fill_value](IteratorValueType& it) { it = fill_value-- % 10 + 1; });

        ::std::size_t max_dis = n;
        if (max_dis)
        {
            *(host_keys.get() + max_dis / 2) = IteratorValueType{/*max_val*/ 777};
            *(host_keys.get() + n - 1) = IteratorValueType{/*2nd max*/ 777};
        }
        host_keys.update_data();

        auto expected_max_offset = ::std::max_element(host_keys.get(), host_keys.get() + n) - host_keys.get();

        auto result_max_offset = std::max_element(CLONE_TEST_POLICY_IDX(exec, 0), first, last) - first;
        wait_and_throw(exec);

        host_keys.retrieve_data();

        EXPECT_EQ(expected_max_offset, result_max_offset, "wrong effect from max_element");
    }
};

DEFINE_TEST(test_minmax_element)
{
    DEFINE_TEST_CONSTRUCTOR(test_minmax_element, 2.0f, 0.80f)

    template <typename Policy, typename Iterator, typename Size>
    void
    operator()(Policy&& exec, Iterator first, Iterator last, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        using IteratorValueType = typename ::std::iterator_traits<Iterator>::value_type;
        auto fill_value = IteratorValueType{ 0 };

        ::std::for_each(host_keys.get(), host_keys.get() + n, [&fill_value](IteratorValueType& it) { it = fill_value++ % 10 + 1; });
        ::std::size_t dis = n;
        if (dis > 1)
        {
            auto min_it = host_keys.get() + /*min_idx*/ dis / 2 - 1;
            *(min_it) = IteratorValueType{/*min_val*/ 0 };

            auto max_it = host_keys.get() + /*max_idx*/ dis / 2;
            *(max_it) = IteratorValueType{/*max_val*/ 777 };
        }
        host_keys.update_data();

        auto expected = ::std::minmax_element(host_keys.get(), host_keys.get() + n);
        auto expected_min = expected.first - host_keys.get();
        auto expected_max = expected.second - host_keys.get();
        ::std::pair<Size, Size> expected_offset = { expected_min, expected_max };

        auto result = std::minmax_element(CLONE_TEST_POLICY_IDX(exec, 0), first, last);
        auto result_min = result.first - first;
        auto result_max = result.second - first;

        wait_and_throw(exec);

        EXPECT_EQ(expected_min, result_min,  "wrong effect from minmax_element: result_min");
        EXPECT_EQ(expected_max, result_max,  "wrong effect from minmax_element: result_max");
    }
};

DEFINE_TEST(test_count)
{
    DEFINE_TEST_CONSTRUCTOR(test_count, 2.0f, 0.80f)

    template <typename Policy, typename Iterator, typename Size>
    void
    operator()(Policy&& exec, Iterator first, Iterator last, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        using ValueType = typename ::std::iterator_traits<Iterator>::value_type;
        using ReturnType = typename ::std::iterator_traits<Iterator>::difference_type;

        ValueType fill_value{0};
        ::std::for_each(host_keys.get(), host_keys.get() + n, [&fill_value](ValueType& value) { value = fill_value++ % 10; });
        host_keys.update_data();

        // check when arbitrary should be counted
        ReturnType expected = (n - 1) / 10 + 1;
        ReturnType result = std::count(CLONE_TEST_POLICY_IDX(exec, 0), first, last, ValueType{0});
        wait_and_throw(exec);

        EXPECT_EQ(expected, result, "wrong effect from count (Test #1 arbitrary to count)");

        // check when none should be counted
        expected = 0;
        result = std::count(CLONE_TEST_POLICY_IDX(exec, 0), first, last, ValueType{12});
        wait_and_throw(exec);

        EXPECT_EQ(expected, result, "wrong effect from count (Test #2 none to count)");

        // check when all should be counted
        ::std::fill(host_keys.get(), host_keys.get() + n, ValueType{7});
        host_keys.update_data();

        expected = n;
        result = std::count(CLONE_TEST_POLICY_IDX(exec, 0), first, last, ValueType{7});
        wait_and_throw(exec);

        EXPECT_EQ(expected, result, "wrong effect from count (Test #3 all to count)");
    }
};

DEFINE_TEST(test_count_if)
{
    DEFINE_TEST_CONSTRUCTOR(test_count_if, 2.0f, 0.80f)

    template <typename Policy, typename Iterator, typename Size>
    void
    operator()(Policy&& exec, Iterator first, Iterator last, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        using ValueType = typename ::std::iterator_traits<Iterator>::value_type;
        using ReturnType = typename ::std::iterator_traits<Iterator>::difference_type;

        ValueType fill_value{0};
        ::std::for_each(host_keys.get(), host_keys.get() + n, [&fill_value](ValueType& value) { value = fill_value++ % 10; });
        host_keys.update_data();

        // check when arbitrary should be counted
        ReturnType expected = (n - 1) / 10 + 1;
        ReturnType result = std::count_if(CLONE_TEST_POLICY_IDX(exec, 0), first, last, TestUtils::IsMultipleOf<ValueType>{10});
        wait_and_throw(exec);

        EXPECT_EQ(expected, result, "wrong effect from count_if (Test #1 arbitrary to count)");

        // check when none should be counted
        expected = 0;
        result = std::count_if(CLONE_TEST_POLICY_IDX(exec, 1), first, last, TestUtils::IsGreatThan<ValueType>{10});
        wait_and_throw(exec);

        EXPECT_EQ(expected, result, "wrong effect from count_if (Test #2 none to count)");

        // check when all should be counted
        expected = n;
        result = std::count_if(CLONE_TEST_POLICY_IDX(exec, 2), first, last, TestUtils::IsLessThan<ValueType>{10});
        wait_and_throw(exec);

        EXPECT_EQ(expected, result, "wrong effect from count_if (Test #3 all to count)");
    }
};

DEFINE_TEST(test_is_partitioned)
{
    DEFINE_TEST_CONSTRUCTOR(test_is_partitioned, 2.0f, 0.80f)

    template <typename Policy, typename Iterator, typename Size>
    void
    operator()(Policy&& exec, Iterator first, Iterator last, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        using ValueType = typename ::std::iterator_traits<Iterator>::value_type;

        if (n < 2)
            return;

        auto less_than = TestUtils::IsLessThan<ValueType>{10};
        auto is_odd = TestUtils::IsOdd<ValueType>{};

        bool expected_bool_less_then = false;
        bool expected_bool_is_odd = false;

        ValueType fill_value{0};
        ::std::for_each(host_keys.get(), host_keys.get() + n, [&fill_value](ValueType& value) { value = ++fill_value; });
        expected_bool_less_then = ::std::is_partitioned(host_keys.get(), host_keys.get() + n, less_than);
        expected_bool_is_odd = ::std::is_partitioned(host_keys.get(), host_keys.get() + n, is_odd);
        host_keys.update_data();

        // check sorted
        bool result_bool = std::is_partitioned(CLONE_TEST_POLICY_IDX(exec, 0), first, last, less_than);
        wait_and_throw(exec);

        EXPECT_EQ(expected_bool_less_then, result_bool, "wrong effect from is_partitioned (Test #1 less than)");

        result_bool = std::is_partitioned(CLONE_TEST_POLICY_IDX(exec, 1), first, last, is_odd);
        wait_and_throw(exec);

        EXPECT_EQ(expected_bool_is_odd, result_bool, "wrong effect from is_partitioned (Test #2 is odd)");

        // The code as below was added to prevent accessor destruction working with host memory
        ::std::partition(host_keys.get(), host_keys.get() + n, is_odd);
        expected_bool_is_odd = ::std::is_partitioned(host_keys.get(), host_keys.get() + n, is_odd);
        host_keys.update_data();

        result_bool = std::is_partitioned(CLONE_TEST_POLICY_IDX(exec, 2), first, last, is_odd);
        wait_and_throw(exec);

        EXPECT_EQ(expected_bool_is_odd, result_bool,
                    "wrong effect from is_partitioned (Test #3 is odd after partition)");
    }
};

DEFINE_TEST(test_transform_reduce_binary)
{
    DEFINE_TEST_CONSTRUCTOR(test_transform_reduce_binary, 2.0f, 0.80f)

    template <typename Policy, typename Iterator1, typename Iterator2, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 /* firs2 */, Iterator2 /* last2 */, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;
        auto value = T1(1);

        ::std::fill(host_keys.get(), host_keys.get() + n, value);
        host_keys.update_data();

        auto result =
            std::transform_reduce(CLONE_TEST_POLICY_IDX(exec, 0), first1, last1, first1, T1(42));
        wait_and_throw(exec);

        EXPECT_TRUE(result == n + 42, "wrong effect from transform_reduce (2 binary)");
    }
};

// TODO: move unique cases to test_lexicographical_compare
DEFINE_TEST(test_lexicographical_compare)
{
    DEFINE_TEST_CONSTRUCTOR(test_lexicographical_compare, 2.0f, 0.80f)

    template <typename Policy, typename Iterator1, typename Iterator2, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 last2, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);
        TestDataTransfer<UDTKind::eVals, Size> host_vals(*this, n);

        using ValueType = typename ::std::iterator_traits<Iterator1>::value_type;

        // INIT
        {
            ValueType fill_value1{0};
            ::std::for_each(host_keys.get(), host_keys.get() + n,
                            [&fill_value1](ValueType& value) { value = fill_value1++ % 10; });
            ValueType fill_value2{0};
            ::std::for_each(host_vals.get(), host_vals.get() + n,
                            [&fill_value2](ValueType& value) { value = fill_value2++ % 10; });
            update_data(host_keys, host_vals);
        }

        auto comp = TestUtils::IsLess<const ValueType&>{};

        // CHECK 1.1: S1 == S2 && len(S1) == len(S2)
        bool is_less_res = std::lexicographical_compare(CLONE_TEST_POLICY_IDX(exec, 0), first1,
                                                          last1, first2, last2, comp);
        wait_and_throw(exec);

        EXPECT_EQ(0, is_less_res, "wrong effect from lex_compare Test 1.1: S1 == S2 && len(S1) == len(S2)");

        // CHECK 1.2: S1 == S2 && len(S1) < len(S2)
        is_less_res = std::lexicographical_compare(CLONE_TEST_POLICY_IDX(exec, 1), first1, last1 - 1,
                                                   first2, last2, comp);
        wait_and_throw(exec);

        EXPECT_EQ(1, is_less_res, "wrong effect from lex_compare Test 1.2: S1 == S2 && len(S1) < len(S2)");

        // CHECK 1.3: S1 == S2 && len(S1) > len(S2)
        is_less_res = std::lexicographical_compare(CLONE_TEST_POLICY_IDX(exec, 2), first1, last1, first2,
                                                   last2 - 1, comp);
        wait_and_throw(exec);

        EXPECT_EQ(0, is_less_res, "wrong effect from lex_compare Test 1.3: S1 == S2 && len(S1) > len(S2)");

        if (n > 1)
        {
            *(host_vals.get() + n - 2) = 222;
            host_vals.update_data();
        }

        // CHECK 2.1: S1 < S2 (PRE-LAST ELEMENT) && len(S1) == len(S2)
        is_less_res = std::lexicographical_compare(CLONE_TEST_POLICY_IDX(exec, 3), first1, last1, first2,
                                                   last2, comp);
        wait_and_throw(exec);

        bool is_less_exp = n > 1 ? 1 : 0;
        EXPECT_EQ(is_less_exp, is_less_res, "wrong effect from lex_compare Test 2.1: S1 < S2 (PRE-LAST) && len(S1) == len(S2)");

        // CHECK 2.2: S1 < S2 (PRE-LAST ELEMENT) && len(S1) > len(S2)
        is_less_res = std::lexicographical_compare(CLONE_TEST_POLICY_IDX(exec, 4), first1, last1, first2,
                                                   last2 - 1, comp);
        wait_and_throw(exec);

        EXPECT_EQ(is_less_exp, is_less_res, "wrong effect from lex_compare Test 2.2: S1 < S2 (PRE-LAST) && len(S1) > len(S2)");

        if (n > 1)
        {
            *(host_keys.get() + n - 2) = 333;
            host_keys.update_data();
        }

        // CHECK 3.1: S1 > S2 (PRE-LAST ELEMENT) && len(S1) == len(S2)
        is_less_res = std::lexicographical_compare(CLONE_TEST_POLICY_IDX(exec, 5), first1, last1, first2,
                                                   last2, comp);
        wait_and_throw(exec);

        EXPECT_EQ(0, is_less_res, "wrong effect from lex_compare Test 3.1: S1 > S2 (PRE-LAST) && len(S1) == len(S2)");

        // CHECK 3.2: S1 > S2 (PRE-LAST ELEMENT) && len(S1) < len(S2)
        is_less_res = std::lexicographical_compare(CLONE_TEST_POLICY_IDX(exec, 6), first1, last1 - 1,
                                                   first2, last2, comp);
        wait_and_throw(exec);

        is_less_exp = n > 1 ? 0 : 1;
        EXPECT_EQ(is_less_exp, is_less_res, "wrong effect from lex_compare Test 3.2: S1 > S2 (PRE-LAST) && len(S1) < len(S2)");
        {
            *host_vals.get() = 444;
            host_vals.update_data();
        }

        // CHECK 4.1: S1 < S2 (FIRST ELEMENT) && len(S1) == len(S2)
        is_less_res = std::lexicographical_compare(CLONE_TEST_POLICY_IDX(exec, 7), first1, last1, first2,
                                                   last2, comp);
        wait_and_throw(exec);

        EXPECT_EQ(1, is_less_res, "wrong effect from lex_compare Test 4.1: S1 < S2 (FIRST) && len(S1) == len(S2)");

        // CHECK 4.2: S1 < S2 (FIRST ELEMENT) && len(S1) > len(S2)
        is_less_res = std::lexicographical_compare(CLONE_TEST_POLICY_IDX(exec, 8), first1, last1, first2,
                                                   last2 - 1, comp);
        wait_and_throw(exec);

        is_less_exp = n > 1 ? 1 : 0;
        EXPECT_EQ(is_less_exp, is_less_res, "wrong effect from lex_compare Test 4.2: S1 < S2 (FIRST) && len(S1) > len(S2)");
        {
            *host_keys.get() = 555;
            host_keys.update_data();
        }

        // CHECK 5.1: S1 > S2 (FIRST ELEMENT) && len(S1) == len(S2)
        is_less_res = std::lexicographical_compare(CLONE_TEST_POLICY_IDX(exec, 9), first1, last1, first2,
                                                   last2, comp);
        wait_and_throw(exec);

        EXPECT_EQ(0, is_less_res, "wrong effect from lex_compare Test 5.1: S1 > S2 (FIRST) && len(S1) == len(S2)");

        // CHECK 5.2: S1 > S2 (FIRST ELEMENT) && len(S1) < len(S2)
        is_less_res = std::lexicographical_compare(CLONE_TEST_POLICY_IDX(exec, 10), first1, last1 - 1,
                                                   first2, last2, comp);
        wait_and_throw(exec);

        is_less_exp = n > 1 ? 0 : 1;
        EXPECT_EQ(is_less_exp, is_less_res, "wrong effect from lex_compare Test 5.2: S1 > S2 (FIRST) && len(S1) < len(S2)");
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
    PRINT_DEBUG("test_reduce");
    test1buffer<alloc_type, test_reduce<ValueType>>();
    PRINT_DEBUG("test_transform_reduce_unary");
    test1buffer<alloc_type, test_transform_reduce_unary<ValueType>>();
    PRINT_DEBUG("test_count");
    test1buffer<alloc_type, test_count<ValueType>>();
    PRINT_DEBUG("test_count_if");
    test1buffer<alloc_type, test_count_if<ValueType>>();
    PRINT_DEBUG("test_is_partitioned");
    test1buffer<alloc_type, test_is_partitioned<ValueType>>();
    PRINT_DEBUG("test_min_element");
    test1buffer<alloc_type, test_min_element<ValueType>>();
    PRINT_DEBUG("test_max_element");
    test1buffer<alloc_type, test_max_element<ValueType>>();
    PRINT_DEBUG("test_minmax_element");
    test1buffer<alloc_type, test_minmax_element<ValueType>>();

    //test2buffers
    PRINT_DEBUG("test_transform_reduce_binary");
    test2buffers<alloc_type, test_transform_reduce_binary<ValueType>>();
    PRINT_DEBUG("test_lexicographical_compare");
    test2buffers<alloc_type, test_lexicographical_compare<ValueType>>();
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
