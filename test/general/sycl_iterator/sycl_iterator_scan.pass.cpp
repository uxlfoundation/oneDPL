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

constexpr int a[] = {0, 0, 1, 1, 2, 6, 6, 9, 9};
constexpr int b[] = {0, 1, 1, 6, 6, 9};
constexpr int c[] = {0, 1, 6, 6, 6, 9, 9};
constexpr int d[] = {7, 7, 7, 8};
constexpr auto a_size = sizeof(a) / sizeof(a[0]);
constexpr auto b_size = sizeof(b) / sizeof(b[0]);
constexpr auto c_size = sizeof(c) / sizeof(c[0]);
constexpr auto d_size = sizeof(d) / sizeof(d[0]);

template <typename Size>
Size
get_size(Size n)
{
    return n + a_size + b_size + c_size + d_size;
}

DEFINE_TEST(test_remove)
{
    DEFINE_TEST_CONSTRUCTOR(test_remove, 2.0f, 0.65f)

    template <typename Policy, typename Iterator, typename Size>
    void
    operator()(Policy&& exec, Iterator first, Iterator last, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        typedef typename ::std::iterator_traits<Iterator>::value_type T1;
        ::std::iota(host_keys.get(), host_keys.get() + n, T1(222));
        host_keys.update_data();

        auto pos = (last - first) / 2;
        auto res1 = ::std::remove(make_new_policy<new_kernel_name<Policy, 0>>(exec), first, last, T1(222 + pos));
        wait_and_throw(exec);

        EXPECT_TRUE(res1 == last - 1, "wrong result from remove");

        host_keys.retrieve_data();
        auto host_first1 = host_keys.get();
        for (int i = 0; i < res1 - first; ++i)
        {
            auto exp = i + 222;
            if (i >= pos)
                ++exp;
            if (host_first1[i] != exp)
                ::std::cout << "Error_1: i = " << i << ", expected " << exp << ", got " << host_first1[i] << ::std::endl;
            EXPECT_TRUE(host_first1[i] == exp, "wrong effect from remove");
        }
    }
};

DEFINE_TEST(test_remove_if)
{
    DEFINE_TEST_CONSTRUCTOR(test_remove_if, 2.0f, 0.65f)

    template <typename T1, typename Size>
    struct CheckState
    {
        Size pos;
        bool operator()(T1 x) const
        {
            return x == T1(222 + pos);
        }
    };

    template <typename Policy, typename Iterator, typename Size>
    void
    operator()(Policy&& exec, Iterator first, Iterator last, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        typedef typename ::std::iterator_traits<Iterator>::value_type T1;

        ::std::iota(host_keys.get(), host_keys.get() + n, T1(222));
        host_keys.update_data();

        auto pos = (last - first) / 2;
        auto res1 = std::remove_if(make_new_policy<new_kernel_name<Policy, 0>>(exec), first, last, CheckState<T1, decltype(pos)>{pos});
        wait_and_throw(exec);

        EXPECT_TRUE(res1 == last - 1, "wrong result from remove_if");

        host_keys.retrieve_data();
        auto host_first1 = host_keys.get();
        for (int i = 0; i < res1 - first; ++i)
        {
            auto exp = i + 222;
            if (i >= pos)
                ++exp;
            if (host_first1[i] != exp)
            {
                ::std::cout << "Error_1: i = " << i << ", expected " << exp << ", got " << host_first1[i] << ::std::endl;
            }
            EXPECT_TRUE(host_first1[i] == exp, "wrong effect from remove_if");
        }
    }
};

DEFINE_TEST(test_unique)
{
    DEFINE_TEST_CONSTRUCTOR(test_unique, 2.0f, 0.65f)

    template <typename IteratorValueType>
    struct IsEq
    {
        bool
        operator()(IteratorValueType a, IteratorValueType b) const
        {
            return a == b;
        }
    };

    template <typename Policy, typename Iterator, typename Size>
    void
    operator()(Policy&& exec, Iterator first, Iterator last, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        using IteratorValueType = typename ::std::iterator_traits<Iterator>::value_type;

        // init
        int index = 0;
        ::std::for_each(host_keys.get(), host_keys.get() + n, [&index](IteratorValueType& value) { value = (index++ + 4) / 4; });
        host_keys.update_data();

        // invoke
        auto result_last = std::unique(make_new_policy<new_kernel_name<Policy, 0>>(exec), first, last, IsEq<IteratorValueType>{});
        wait_and_throw(exec);

        auto result_size = result_last - first;

        std::int64_t expected_size = (n - 1) / 4 + 1;

        // check
        bool is_correct = result_size == expected_size;
#if _ONEDPL_DEBUG_SYCL
        if (!is_correct)
            ::std::cout << "buffer size: got " << result_last - first << ", expected " << expected_size << ::std::endl;
#endif // _ONEDPL_DEBUG_SYCL

        host_keys.retrieve_data();
        auto host_first1 = host_keys.get();
        for (int i = 0; i < ::std::min(result_size, expected_size) && is_correct; ++i)
        {
            if (*(host_first1 + i) != i + 1)
            {
                is_correct = false;
#if _ONEDPL_DEBUG_SYCL
                ::std::cout << "got: " << *(host_first1 + i) << "[" << i << "], "
                          << "expected: " << i + 1 << "[" << i << "]" << ::std::endl;
#endif // _ONEDPL_DEBUG_SYCL
            }
            EXPECT_TRUE(is_correct, "wrong effect from unique");
        }
    }
};

DEFINE_TEST(test_partition)
{
    DEFINE_TEST_CONSTRUCTOR(test_partition, 2.0f, 0.65f)

    template <typename IteratorValueType>
    struct IsMultipleOf3And2
    {
        bool operator()(IteratorValueType value) const
        {
            return (value % 3 == 0) && (value % 2 == 0);
        }
    };

    template <typename Policy, typename Iterator, typename Size>
    void
    operator()(Policy&& exec, Iterator first, Iterator last, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        using IteratorValueType = typename ::std::iterator_traits<Iterator>::value_type;

        // init
        ::std::iota(host_keys.get(), host_keys.get() + n, IteratorValueType{ 0 });
        host_keys.update_data();

        // invoke partition
        auto unary_op = IsMultipleOf3And2<IteratorValueType>{};
        auto res = ::std::partition(make_new_policy<new_kernel_name<Policy, 0>>(exec), first, last, unary_op);
        wait_and_throw(exec);

        // check
        host_keys.retrieve_data();
        EXPECT_TRUE(::std::all_of(host_keys.get(), host_keys.get() + (res - first), unary_op) &&
                        !::std::any_of(host_keys.get() + (res - first), host_keys.get() + n, unary_op),
                    "wrong effect from partition");
        // init
        ::std::iota(host_keys.get(), host_keys.get() + n, IteratorValueType{0});
        host_keys.update_data();

        // invoke stable_partition
        res = ::std::stable_partition(make_new_policy<new_kernel_name<Policy, 1>>(exec), first, last, unary_op);
        wait_and_throw(exec);

        host_keys.retrieve_data();
        EXPECT_TRUE(::std::all_of(host_keys.get(), host_keys.get() + (res - first), unary_op) &&
                        !::std::any_of(host_keys.get() + (res - first), host_keys.get() + n, unary_op) &&
                        ::std::is_sorted(host_keys.get(), host_keys.get() + (res - first)) &&
                        ::std::is_sorted(host_keys.get() + (res - first), host_keys.get() + n),
                    "wrong effect from stable_partition");
    }
};

DEFINE_TEST(test_transform_inclusive_scan)
{
    DEFINE_TEST_CONSTRUCTOR(test_transform_inclusive_scan, 2.0f, 0.65f)

    template <typename T1>
    struct Mult2
    {
        T1
        operator()(T1 x) const
        {
            return x * 2;
        }
    };

    template <typename T1>
    struct TransformOp
    {
        T1
        operator()(T1 x) const
        {
            return x * 2;
        }
    };

    template <typename Policy, typename Iterator1, typename Iterator2, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 last2, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);
        TestDataTransfer<UDTKind::eVals, Size> host_vals(*this, n);

        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;
        auto value = T1(333);

        ::std::fill(host_keys.get(), host_keys.get() + n, T1(1));
        host_keys.update_data();

        auto res1 = std::transform_inclusive_scan(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, last1, first2, std::plus<T1>(), Mult2<T1>{}, value);
        wait_and_throw(exec);

        EXPECT_TRUE(res1 == last2, "wrong result from transform_inclusive_scan_1");

        retrieve_data(host_keys, host_vals);

        T1 ii = value;
        for (int i = 0; i < last2 - first2; ++i)
        {
            ii += 2 * host_keys.get()[i];
            if (host_vals.get()[i] != ii)
            {
                ::std::cout << "Error in scan_1: i = " << i << ", expected " << ii << ", got " << host_vals.get()[i]
                            << ::std::endl;
            }
            EXPECT_TRUE(host_vals.get()[i] == ii, "wrong effect from transform_inclusive_scan_1");
        }

        // without initial value
        auto res2 = std::transform_inclusive_scan(make_new_policy<new_kernel_name<Policy, 1>>(exec), first1, last1, first2, std::plus<T1>(), TransformOp<T1>{});
        EXPECT_TRUE(res2 == last2, "wrong result from transform_inclusive_scan_2");

        retrieve_data(host_keys, host_vals);

        ii = 0;
        for (int i = 0; i < last2 - first2; ++i)
        {
            ii += 2 * host_keys.get()[i];
            if (host_vals.get()[i] != ii)
            {
                ::std::cout << "Error in scan_2: i = " << i << ", expected " << ii << ", got " << host_vals.get()[i]
                            << ::std::endl;
            }
            EXPECT_TRUE(host_vals.get()[i] == ii, "wrong effect from transform_inclusive_scan_2");
        }
    }
};

DEFINE_TEST(test_transform_exclusive_scan)
{
    DEFINE_TEST_CONSTRUCTOR(test_transform_exclusive_scan, 2.0f, 0.65f)

    template <typename T1>
    struct TransformOp
    {
        T1 operator()(T1 x) const
        {
            return x * 2;
        }
    };

    template <typename Policy, typename Iterator1, typename Iterator2, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 last2, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);
        TestDataTransfer<UDTKind::eVals, Size> host_vals(*this, n);

        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;

        ::std::fill(host_keys.get(), host_keys.get() + n, T1(1));
        host_keys.update_data();

        auto res1 = std::transform_exclusive_scan(make_new_policy<new_kernel_name<Policy, 2>>(exec), first1, last1, first2, T1{}, std::plus<T1>(), TransformOp<T1>{});
        wait_and_throw(exec);

        EXPECT_TRUE(res1 == last2, "wrong result from transform_exclusive_scan");

        auto ii = T1(0);

        retrieve_data(host_keys, host_vals);

        for (size_t i = 0; i < last2 - first2; ++i)
        {
            if (host_vals.get()[i] != ii)
                ::std::cout << "Error: i = " << i << ", expected " << ii << ", got " << host_vals.get()[i] << ::std::endl;

            //EXPECT_TRUE(host_vals.get()[i] == ii, "wrong effect from transform_exclusive_scan");
            ii += 2 * host_keys.get()[i];
        }
    }
};

DEFINE_TEST(test_copy_if)
{
    DEFINE_TEST_CONSTRUCTOR(test_copy_if, 2.0f, 0.65f)

    template <typename T1>
    struct GreatThenMinusOne
    {
        bool
        operator()(T1 x) const
        {
            return x > -1;
        }
    };

    template <typename Policy, typename Iterator1, typename Iterator2, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 last2, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);
        TestDataTransfer<UDTKind::eVals, Size> host_vals(*this, n);

        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;

        ::std::iota(host_keys.get(), host_keys.get() + n, T1(222));
        host_keys.update_data();

        auto res1 = std::copy_if(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, last1, first2, GreatThenMinusOne<T1>{});
        wait_and_throw(exec);

        EXPECT_TRUE(res1 == last2, "wrong result from copy_if_1");

        host_vals.retrieve_data();
        auto host_first2 = host_vals.get();
        for (int i = 0; i < res1 - first2; ++i)
        {
            auto exp = i + 222;
            if (host_first2[i] != exp)
            {
                ::std::cout << "Error_1: i = " << i << ", expected " << exp << ", got " << host_first2[i] << ::std::endl;
            }
            EXPECT_TRUE(host_first2[i] == exp, "wrong effect from copy_if_1");
        }

        auto res2 = std::copy_if(make_new_policy<new_kernel_name<Policy, 1>>(exec), first1, last1, first2, TestUtils::IsOdd<T1>{});
        wait_and_throw(exec);

        EXPECT_TRUE(res2 == first2 + (last2 - first2) / 2, "wrong result from copy_if_2");

        host_vals.retrieve_data();
        host_first2 = host_vals.get();
        for (int i = 0; i < res2 - first2; ++i)
        {
            auto exp = 2 * i + 1 + 222;
            if (host_first2[i] != exp)
            {
                ::std::cout << "Error_2: i = " << i << ", expected " << exp << ", got " << host_first2[i] << ::std::endl;
            }
            EXPECT_TRUE(host_first2[i] == exp, "wrong effect from copy_if_2");
        }
    }
};

DEFINE_TEST(test_unique_copy)
{
    DEFINE_TEST_CONSTRUCTOR(test_unique_copy, 2.0f, 0.65f)

    template <typename Iterator1ValueType>
    struct IsEq
    {
        bool
        operator()(Iterator1ValueType a, Iterator1ValueType b) const
        {
            return a == b;
        }
    };

    template <typename Policy, typename Iterator1, typename Iterator2, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 /* last2 */, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);
        TestDataTransfer<UDTKind::eVals, Size> host_vals(*this, n);

        using Iterator1ValueType = typename ::std::iterator_traits<Iterator1>::value_type;

        // init
        int index = 0;
        ::std::for_each(host_keys.get(), host_keys.get() + n, [&index](Iterator1ValueType& value) { value = (index++ + 4) / 4; });
        ::std::fill(host_vals.get(), host_vals.get() + n, Iterator1ValueType{ -1 });
        update_data(host_keys, host_vals);

        // invoke
        auto f = IsEq<Iterator1ValueType>{};
        auto result_first = first2;
        auto result_last =
            ::std::unique_copy(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, last1, result_first, f);
        wait_and_throw(exec);

        auto result_size = result_last - result_first;

        std::int64_t expected_size = (n - 1) / 4 + 1;

        // check
        bool is_correct = result_size == expected_size;
#if _ONEDPL_DEBUG_SYCL
        if (!is_correct)
            ::std::cout << "buffer size: got " << result_last - result_first << ", expected " << expected_size
                      << ::std::endl;
#endif // _ONEDPL_DEBUG_SYCL

        host_vals.retrieve_data();
        auto host_first2 = host_vals.get();
        for (int i = 0; i < ::std::min(result_size, expected_size) && is_correct; ++i)
        {
            if (*(host_first2 + i) != i + 1)
            {
                is_correct = false;
                ::std::cout << "got: " << *(host_first2 + i) << "[" << i << "], "
                          << "expected: " << i + 1 << "[" << i << "]" << ::std::endl;
            }
            EXPECT_TRUE(is_correct, "wrong effect from unique_copy");
        }
    }
};

DEFINE_TEST(test_partition_copy)
{
    DEFINE_TEST_CONSTRUCTOR(test_partition_copy, 2.0f, 0.65f)

    template <typename Iterator1ValueType>
    struct IsMultipleOf3And2
    {
        template <typename T1>
        bool
        operator()(T1 x) const
        {
            return (x % 3 == 0) && (x % 2 == 0);
        }
    };

    template <typename Policy, typename Iterator1, typename Iterator2, typename Iterator3, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 /* last2 */, Iterator3 first3,
               Iterator3 /* last3 */, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);
        TestDataTransfer<UDTKind::eVals, Size> host_vals(*this, n);
        TestDataTransfer<UDTKind::eRes,  Size> host_res (*this, n);

        using Iterator1ValueType = typename ::std::iterator_traits<Iterator1>::value_type;
        using Iterator2ValueType = typename ::std::iterator_traits<Iterator2>::value_type;
        using Iterator3ValueType = typename ::std::iterator_traits<Iterator3>::value_type;
        auto f = IsMultipleOf3And2<Iterator1ValueType>{};

        // init
        ::std::iota(host_keys.get(), host_keys.get() + n, Iterator1ValueType{0});
        ::std::fill(host_vals.get(), host_vals.get() + n, Iterator2ValueType{-1});
        ::std::fill(host_res.get(),   host_res.get() + n, Iterator3ValueType{-2});
        update_data(host_keys, host_vals, host_res);

        // invoke
        auto res =
            ::std::partition_copy(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, last1, first2, first3, f);
        wait_and_throw(exec);

        retrieve_data(host_keys, host_vals, host_res);

        // init for expected
        ::std::vector<Iterator2ValueType> exp_true(n, -1);
        ::std::vector<Iterator3ValueType> exp_false(n, -2);
        auto exp_true_first = exp_true.begin();
        auto exp_false_first = exp_false.begin();

        // invoke for expected
        auto exp = ::std::partition_copy(host_keys.get(), host_keys.get() + n, exp_true_first, exp_false_first, f);

        // check
        bool is_correct = (exp.first - exp_true_first) == (res.first - first2) &&
                          (exp.second - exp_false_first) == (res.second - first3);
#if _ONEDPL_DEBUG_SYCL
        if (!is_correct)
            ::std::cout << "N =" << n << ::std::endl
                      << "buffer size: got {" << res.first - first2 << "," << res.second - first3 << "}, expected {"
                      << exp.first - exp_true_first << "," << exp.second - exp_false_first << "}" << ::std::endl;
#endif // _ONEDPL_DEBUG_SYCL

        for (int i = 0; i < ::std::min(exp.first - exp_true_first, res.first - first2) && is_correct; ++i)
        {
            if (*(exp_true_first + i) != *(host_vals.get() + i))
            {
                is_correct = false;
#if _ONEDPL_DEBUG_SYCL
                ::std::cout << "TRUE> got: " << *(host_vals.get() + i) << "[" << i << "], "
                          << "expected: " << *(exp_true_first + i) << "[" << i << "]" << ::std::endl;
#endif // _ONEDPL_DEBUG_SYCL
            }
        }

        for (int i = 0; i < ::std::min(exp.second - exp_false_first, res.second - first3) && is_correct; ++i)
        {
            if (*(exp_false_first + i) != *(host_res.get() + i))
            {
                is_correct = false;
#if _ONEDPL_DEBUG_SYCL
                ::std::cout << "FALSE> got: " << *(host_res.get() + i) << "[" << i << "], "
                          << "expected: " << *(exp_false_first + i) << "[" << i << "]" << ::std::endl;
#endif // _ONEDPL_DEBUG_SYCL
            }
        }

        EXPECT_TRUE(is_correct, "wrong effect from partition_copy");
    }
};

DEFINE_TEST(test_set_intersection)
{
    DEFINE_TEST_CONSTRUCTOR(test_set_intersection, 2.0f, 0.65f)

    template <typename Policy, typename Iterator1, typename Iterator2, typename Iterator3, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 last2, Iterator3 first3,
               Iterator3 last3, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, get_size(n));
        TestDataTransfer<UDTKind::eVals, Size> host_vals(*this, get_size(n));
        TestDataTransfer<UDTKind::eRes,  Size> host_res (*this, get_size(n));

        //first test case
        last1 = first1 + a_size;
        last2 = first2 + b_size;
        ::std::copy(a, a + a_size, host_keys.get());
        ::std::copy(b, b + b_size, host_vals.get());
        host_keys.update_data(a_size);
        host_vals.update_data(b_size);

        last3 = ::std::set_intersection(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, last1, first2, last2,
                                      first3);
        wait_and_throw(exec);

        host_res.retrieve_data();
        auto nres = last3 - first3;

        EXPECT_TRUE(nres == 6, "wrong size of intersection of a, b");

        auto result = ::std::includes(host_keys.get(), host_keys.get() + a_size, host_res.get(), host_res.get() + nres) &&
                      ::std::includes(host_vals.get(), host_vals.get() + b_size, host_res.get(), host_res.get() + nres);
        wait_and_throw(exec);

        EXPECT_TRUE(result, "wrong effect from set_intersection a, b");

        { //second test case

            last2 = first2 + d_size;
            ::std::copy(a, a + a_size, host_keys.get());
            ::std::copy(d, d + d_size, host_vals.get());
            host_keys.update_data(a_size);
            host_vals.update_data(b_size);

            last3 = ::std::set_intersection(make_new_policy<new_kernel_name<Policy, 1>>(exec), first1, last1, first2,
                                          last2, first3);
            wait_and_throw(exec);

            auto nres = last3 - first3;
            EXPECT_TRUE(nres == 0, "wrong size of intersection of a, d");
        }
    }
};

DEFINE_TEST(test_set_difference)
{
    DEFINE_TEST_CONSTRUCTOR(test_set_difference, 2.0f, 0.65f)

    template <typename Policy, typename Iterator1, typename Iterator2, typename Iterator3, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 last2, Iterator3 first3,
               Iterator3 last3, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, get_size(n));
        TestDataTransfer<UDTKind::eVals, Size> host_vals(*this, get_size(n));
        TestDataTransfer<UDTKind::eRes,  Size> host_res (*this, get_size(n));

        last1 = first1 + a_size;
        last2 = first2 + b_size;

        ::std::copy(a, a + a_size, host_keys.get());
        ::std::copy(b, b + b_size, host_vals.get());
        host_keys.update_data(a_size);
        host_vals.update_data(b_size);

        last3 = ::std::set_difference(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, last1, first2, last2, first3);
        wait_and_throw(exec);

        int res_expect[a_size];
        host_res.retrieve_data();
        auto nres_expect = ::std::set_difference(host_keys.get(), host_keys.get() + a_size, host_vals.get(), host_vals.get() + b_size, res_expect) - res_expect;
        EXPECT_EQ_N(host_res.get(), res_expect, nres_expect, "wrong effect from set_difference a, b");
    }
};

DEFINE_TEST(test_set_union)
{
    DEFINE_TEST_CONSTRUCTOR(test_set_union, 2.0f, 0.65f)

    template <typename Policy, typename Iterator1, typename Iterator2, typename Iterator3, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 last2, Iterator3 first3,
               Iterator3 last3, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, get_size(n));
        TestDataTransfer<UDTKind::eVals, Size> host_vals(*this, get_size(n));
        TestDataTransfer<UDTKind::eRes,  Size> host_res (*this, get_size(n));

        last1 = first1 + a_size;
        last2 = first2 + b_size;

        ::std::copy(a, a + a_size, host_keys.get());
        ::std::copy(b, b + b_size, host_vals.get());
        host_keys.update_data(a_size);
        host_vals.update_data(b_size);

        last3 = ::std::set_union(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, last1, first2, last2, first3);
        wait_and_throw(exec);

        int res_expect[a_size + b_size];
        host_res.retrieve_data();
        auto nres_expect =
            ::std::set_union(host_keys.get(), host_keys.get() + a_size, host_vals.get(), host_vals.get() + b_size, res_expect) - res_expect;
        EXPECT_EQ_N(host_res.get(), res_expect, nres_expect, "wrong effect from set_union a, b");
    }
};

DEFINE_TEST(test_set_symmetric_difference)
{
    DEFINE_TEST_CONSTRUCTOR(test_set_symmetric_difference, 2.0f, 0.65f)

    template <typename Policy, typename Iterator1, typename Iterator2, typename Iterator3, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 last2, Iterator3 first3,
               Iterator3 last3, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, get_size(n));
        TestDataTransfer<UDTKind::eVals, Size> host_vals(*this, get_size(n));
        TestDataTransfer<UDTKind::eRes, Size>  host_res (*this, get_size(n));

        last1 = first1 + a_size;
        last2 = first2 + b_size;

        ::std::copy(a, a + a_size, host_keys.get());
        ::std::copy(b, b + b_size, host_vals.get());
        host_keys.update_data(a_size);
        host_vals.update_data(b_size);

        last3 = ::std::set_symmetric_difference(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, last1,
                                                first2, last2, first3);
        wait_and_throw(exec);

        int res_expect[a_size + b_size];
        retrieve_data(host_keys, host_vals, host_res);
        auto nres_expect = ::std::set_symmetric_difference(host_keys.get(), host_keys.get() + a_size, host_vals.get(),
                                                           host_vals.get() + b_size, res_expect) -
                           res_expect;
        EXPECT_EQ_N(host_res.get(), res_expect, nres_expect, "wrong effect from set_symmetric_difference a, b");
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
    PRINT_DEBUG("test_partition");
    test1buffer<alloc_type, test_partition<ValueType>>();
    PRINT_DEBUG("test_remove");
    test1buffer<alloc_type, test_remove<ValueType>>();
    PRINT_DEBUG("test_remove_if");
    test1buffer<alloc_type, test_remove_if<ValueType>>();
    PRINT_DEBUG("test_unique");
    test1buffer<alloc_type, test_unique<ValueType>>();

    //test2buffers
    PRINT_DEBUG("test_transform_inclusive_scan");
    test2buffers<alloc_type, test_transform_inclusive_scan<ValueType>>();
    PRINT_DEBUG("test_transform_exclusive_scan");
    test2buffers<alloc_type, test_transform_exclusive_scan<ValueType>>();
    PRINT_DEBUG("test_copy_if");
    test2buffers<alloc_type, test_copy_if<ValueType>>();
    PRINT_DEBUG("test_unique_copy");
    test2buffers<alloc_type, test_unique_copy<ValueType>>();

    //test3buffers
    PRINT_DEBUG("test_partition_copy");
    test3buffers<alloc_type, test_partition_copy<ValueType>>();
    PRINT_DEBUG("test_set_symmetric_difference");
    test3buffers<alloc_type, test_set_symmetric_difference<ValueType>>();
    PRINT_DEBUG("test_set_union");
    test3buffers<alloc_type, test_set_union<ValueType>>();
    PRINT_DEBUG("test_set_difference");
    test3buffers<alloc_type, test_set_difference<ValueType>>();
    PRINT_DEBUG("test_set_intersection");
    test3buffers<alloc_type, test_set_intersection<ValueType>>();
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
