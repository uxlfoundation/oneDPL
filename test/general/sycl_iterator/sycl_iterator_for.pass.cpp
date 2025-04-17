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

struct Inc
{
    template <typename T>
    void
    operator()(T& x) const
    {
        ++x;
    }
};

struct Flip
{
    std::int32_t val;
    Flip(std::int32_t y) : val(y) {}
    template <typename T>
    T
    operator()(const T& x) const
    {
        return val - x;
    }
};

template <typename T>
struct Generator_count
{
    T def_val;
    Generator_count(const T& val) : def_val(val) {}
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

// created just to check destroy and destroy_n correctness
template <typename T>
struct SyclTypeWrapper
{
    T __value;

    explicit SyclTypeWrapper(const T& value = T{4}) : __value(value) {}
    ~SyclTypeWrapper() { __value = -2; }
    bool
    operator==(const SyclTypeWrapper& other) const
    {
        return __value == other.__value;
    }
};

// this wrapper is needed to take into account not only kernel name,
// but also other types (for example, iterator's value type)
template <typename... T>
struct policy_name_wrapper
{
};

DEFINE_TEST(test_uninitialized_fill)
{
    DEFINE_TEST_CONSTRUCTOR(test_uninitialized_fill, 1.0f, 1.0f)

    template <typename Policy, typename Iterator1, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 /* last1 */, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;
        auto value = T1(2);

        ::std::uninitialized_fill(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1 + (n / 3), first1 + (n / 2),
                                  value);
        wait_and_throw(exec);

        host_keys.retrieve_data();
        EXPECT_TRUE(check_values(host_keys.get() + (n / 3), host_keys.get() + (n / 2),
                                 value),
                    "wrong effect from uninitialized_fill");
    }
};

DEFINE_TEST(test_uninitialized_fill_n)
{
    DEFINE_TEST_CONSTRUCTOR(test_uninitialized_fill_n, 1.0f, 1.0f)

    template <typename Policy, typename Iterator1, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 /* last1 */, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;
        auto value = T1(2);

        ::std::uninitialized_fill_n(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, n, value + 1);
        wait_and_throw(exec);

        host_keys.retrieve_data();
        EXPECT_TRUE(check_values(host_keys.get(), host_keys.get() + n, value + 1),
                    "wrong effect from uninitialized_fill_n");
    }
};

DEFINE_TEST(test_uninitialized_default_construct)
{
    DEFINE_TEST_CONSTRUCTOR(test_uninitialized_default_construct, 1.0f, 1.0f)

    template <typename Policy, typename Iterator1, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 /* last1 */, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;
        auto value = T1{ 2 };

        T1 exp_value; // default-constructed value
        ::std::fill(host_keys.get(), host_keys.get() + n, value);
        host_keys.update_data();

        ::std::uninitialized_default_construct(make_new_policy<new_kernel_name<Policy, 0>>(exec),
                                             first1 + (n / 3), first1 + (n / 2));
        wait_and_throw(exec);

        host_keys.retrieve_data();
        EXPECT_TRUE(check_values(host_keys.get() + (n / 3), host_keys.get() + (n / 2), exp_value),
                    "wrong effect from uninitialized_default_construct");
    }
};

DEFINE_TEST(test_uninitialized_default_construct_n)
{
    DEFINE_TEST_CONSTRUCTOR(test_uninitialized_default_construct_n, 1.0f, 1.0f)

    template <typename Policy, typename Iterator1, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 /* last1 */, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;
        auto value = T1{ 2 };

        T1 exp_value; // default-constructed value
        ::std::fill(host_keys.get(), host_keys.get() + n, value);
        host_keys.update_data();

        ::std::uninitialized_default_construct_n(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, n);
        wait_and_throw(exec);

        host_keys.retrieve_data();
        EXPECT_TRUE(check_values(host_keys.get(), host_keys.get() + n, exp_value),
                    "wrong effect from uninitialized_default_construct_n");
    }
};

DEFINE_TEST(test_uninitialized_value_construct)
{
    DEFINE_TEST_CONSTRUCTOR(test_uninitialized_value_construct, 1.0f, 1.0f)

    template <typename Policy, typename Iterator1, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 /* last1 */, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;
        auto value = T1(2);
        ::std::fill(host_keys.get(), host_keys.get() + n, value);
        host_keys.update_data();

        ::std::uninitialized_value_construct(make_new_policy<new_kernel_name<Policy, 0>>(exec),
                                           first1 + (n / 3), first1 + (n / 2));
        wait_and_throw(exec);

        host_keys.retrieve_data();
        EXPECT_TRUE(check_values(host_keys.get() + (n / 3), host_keys.get() + (n / 2), T1{}),
                    "wrong effect from uninitialized_value_construct");
    }
};

DEFINE_TEST(test_uninitialized_value_construct_n)
{
    DEFINE_TEST_CONSTRUCTOR(test_uninitialized_value_construct_n, 1.0f, 1.0f)

    template <typename Policy, typename Iterator1, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 /* last1 */, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;
        auto value = T1(2);

        ::std::fill(host_keys.get(), host_keys.get() + n, value);
        host_keys.update_data();

        ::std::uninitialized_value_construct_n(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, n);
        wait_and_throw(exec);

        host_keys.retrieve_data();
        EXPECT_TRUE(check_values(host_keys.get(), host_keys.get() + n, T1{}),
                    "wrong effect from uninitialized_value_construct_n");
    }
};

DEFINE_TEST(test_destroy)
{
    DEFINE_TEST_CONSTRUCTOR(test_destroy, 1.0f, 1.0f)

    template <typename Policy, typename Iterator1, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 /* last1 */, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;
        auto value = T1{ 2 };
        ::std::fill(host_keys.get(), host_keys.get() + n, value);
        host_keys.update_data();

        ::std::destroy(make_new_policy<policy_name_wrapper<new_kernel_name<Policy, 0>, T1>>(exec), first1 + (n / 3),
                       first1 + (n / 2));
        if (!::std::is_trivially_destructible_v<T1>)
            value = T1{-2};
        wait_and_throw(exec);

        host_keys.retrieve_data();
        EXPECT_TRUE(check_values(host_keys.get() + (n / 3), host_keys.get() + (n / 2), value),
                    "wrong effect from destroy");
    }
};

DEFINE_TEST(test_destroy_n)
{
    DEFINE_TEST_CONSTRUCTOR(test_destroy_n, 1.0f, 1.0f)

    template <typename Policy, typename Iterator1, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 /* last1 */, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;
        auto value = T1{ 2 };

        ::std::fill(host_keys.get(), host_keys.get() + n, value);
        host_keys.update_data();

        ::std::destroy_n(make_new_policy<policy_name_wrapper<new_kernel_name<Policy, 0>, T1>>(exec), first1, n);
        if(!::std::is_trivially_destructible_v<T1>)
            value = T1{-2};
        wait_and_throw(exec);

        host_keys.retrieve_data();
        EXPECT_TRUE(check_values(host_keys.get(), host_keys.get() + n, value),
                    "wrong effect from destroy_n");
    }
};

DEFINE_TEST(test_fill)
{
    DEFINE_TEST_CONSTRUCTOR(test_fill, 1.0f, 1.0f)

    template <typename Policy, typename Iterator1, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 /* last1 */, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;
        auto value = T1(2);

        ::std::fill(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1 + (n / 3), first1 + (n / 2), value);
        wait_and_throw(exec);

        host_keys.retrieve_data();
        EXPECT_TRUE(check_values(host_keys.get() + (n / 3), host_keys.get() + (n / 2), value), "wrong effect from fill");
    }
};

DEFINE_TEST(test_fill_n)
{
    DEFINE_TEST_CONSTRUCTOR(test_fill_n, 1.0f, 1.0f)

    template <typename Policy, typename Iterator1, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 /* last1 */, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;
        auto value = T1(2);

        ::std::fill_n(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, n, value + 1);
        wait_and_throw(exec);

        host_keys.retrieve_data();
        EXPECT_TRUE(check_values(host_keys.get(), host_keys.get() + n, value + 1), "wrong effect from fill_n");
    }
};

DEFINE_TEST(test_generate)
{
    DEFINE_TEST_CONSTRUCTOR(test_generate, 1.0f, 1.0f)

    template <typename Policy, typename Iterator1, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 /* last1 */, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;
        auto value = T1(4);

        ::std::generate(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1 + (n / 3), first1 + (n / 2),
                      Generator_count<T1>(value));
        wait_and_throw(exec);

        host_keys.retrieve_data();
        EXPECT_TRUE(check_values(host_keys.get() + (n / 3), host_keys.get() + (n / 2), value),
                    "wrong effect from generate");
    }
};

DEFINE_TEST(test_generate_n)
{
    DEFINE_TEST_CONSTRUCTOR(test_generate_n, 1.0f, 1.0f)

    template <typename Policy, typename Iterator1, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 /* last1 */, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;
        auto value = T1(4);

        ::std::generate_n(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, n, Generator_count<T1>(value + 1));
        wait_and_throw(exec);

        host_keys.retrieve_data();
        EXPECT_TRUE(check_values(host_keys.get(), host_keys.get() + n, value + 1),
                    "wrong effect from generate_n");
    }
};

DEFINE_TEST(test_for_each)
{
    DEFINE_TEST_CONSTRUCTOR(test_for_each, 1.0f, 1.0f)

    template <typename Policy, typename Iterator1, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 /* last1 */, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;
        auto value = T1(6);

        ::std::fill(host_keys.get(), host_keys.get() + n, value);
        ::std::fill(host_keys.get() + (n / 3), host_keys.get() + (n / 2), value - 1);
        host_keys.update_data();

        ::std::for_each(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1 + (n / 3), first1 + (n / 2), Inc());
        wait_and_throw(exec);

        // We call due to SYCL 1.2.1: 4.7.2.3.
        // If the host memory is modified by the host,
        // or mapped to another buffer or image during the lifetime of this buffer,
        // then the results are undefined
        host_keys.retrieve_data();
        EXPECT_TRUE(check_values(host_keys.get(), host_keys.get() + n, value), "wrong effect from for_each");
    }
};

DEFINE_TEST(test_for_each_n)
{
    DEFINE_TEST_CONSTRUCTOR(test_for_each_n, 1.0f, 1.0f)

    template <typename Policy, typename Iterator1, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 /* last1 */, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;
        auto value = T1(6);

        ::std::fill(host_keys.get(), host_keys.get() + n, value);
        host_keys.update_data();

        ::std::for_each_n(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, n, Inc());
        wait_and_throw(exec);

        host_keys.retrieve_data();
        EXPECT_TRUE(check_values(host_keys.get(), host_keys.get() + n, value + 1),
                    "wrong effect from for_each_n");
    }
};

DEFINE_TEST(test_replace)
{
    DEFINE_TEST_CONSTRUCTOR(test_replace, 1.0f, 1.0f)

    template <typename Policy, typename Iterator1, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;
        auto value = T1(5);

        ::std::fill(host_keys.get(), host_keys.get() + n, value);
        host_keys.update_data();

        ::std::replace(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, last1, value, T1(value + 1));
        wait_and_throw(exec);

        host_keys.retrieve_data();
        EXPECT_TRUE(check_values(host_keys.get(), host_keys.get() + n, value + 1),
                    "wrong effect from replace");
    }
};

DEFINE_TEST(test_replace_if)
{
    DEFINE_TEST_CONSTRUCTOR(test_replace_if, 1.0f, 1.0f)

    template <typename Policy, typename Iterator1, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;
        auto value = T1(6);
        ::std::fill(host_keys.get(), host_keys.get() + n, value);
        host_keys.update_data();

        ::std::replace_if(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, last1,
                          oneapi::dpl::__internal::__equal_value<T1>(value), T1(value + 1));
        wait_and_throw(exec);

        host_keys.retrieve_data();
        EXPECT_TRUE(check_values(host_keys.get(), host_keys.get() + n, value + 1),
                    "wrong effect from replace_if");
    }
};

DEFINE_TEST(test_reverse)
{
    DEFINE_TEST_CONSTRUCTOR(test_reverse, 1.0f, 1.0f)

    template <typename Policy, typename Iterator1, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first, Iterator1 last, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);
        host_keys.retrieve_data();

        using IteratorValyeType = typename ::std::iterator_traits<Iterator1>::value_type;

        ::std::vector<IteratorValyeType> local_copy(n);
        local_copy.assign(host_keys.get(), host_keys.get() + n);
        ::std::reverse(local_copy.begin(), local_copy.end());

        ::std::reverse(make_new_policy<new_kernel_name<Policy, 0>>(exec), first, last);
        wait_and_throw(exec);

        host_keys.retrieve_data();
        auto host_first1 = host_keys.get();
        for (int i = 0; i < (last - first); ++i)
            EXPECT_TRUE(local_copy[i] == host_first1[i], "wrong effect from reverse");
    }
};

DEFINE_TEST(test_rotate)
{
    DEFINE_TEST_CONSTRUCTOR(test_rotate, 1.0f, 1.0f)

    template <typename Policy, typename Iterator1, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first, Iterator1 last, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);
        host_keys.retrieve_data();

        using IteratorValyeType = typename ::std::iterator_traits<Iterator1>::value_type;

        ::std::vector<IteratorValyeType> local_copy(n);
        local_copy.assign(host_keys.get(), host_keys.get() + n);
        ::std::rotate(local_copy.begin(), local_copy.begin() + 1, local_copy.end());

        ::std::rotate(make_new_policy<new_kernel_name<Policy, 0>>(exec), first, first + 1, last);
        wait_and_throw(exec);

        host_keys.retrieve_data();
        auto host_first1 = host_keys.get();
        for (int i = 0; i < (last - first); ++i)
            EXPECT_TRUE(local_copy[i] == host_first1[i], "wrong effect from rotate");
    }
};

DEFINE_TEST(test_includes)
{
    DEFINE_TEST_CONSTRUCTOR(test_includes, 1.0f, 1.0f)

    template <typename Policy, typename Iterator1, typename Iterator2, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 last2, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, get_size(n));
        TestDataTransfer<UDTKind::eVals, Size> host_vals(*this, get_size(n));

        //first test case
        last1 = first1 + a_size;
        last2 = first2 + b_size;

        ::std::copy(a, a + a_size, host_keys.get());
        ::std::copy(b, b + b_size, host_vals.get());
        host_keys.update_data(a_size);
        host_vals.update_data(b_size);

        auto result = ::std::includes(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, last1, first2, last2);
        wait_and_throw(exec);

        EXPECT_TRUE(result, "wrong effect from includes a, b");

        host_vals.retrieve_data();
        ::std::copy(c, c + c_size, host_vals.get());
        host_vals.update_data(c_size);

        result = ::std::includes(make_new_policy<new_kernel_name<Policy, 1>>(exec), first1, last1, first2, last2);
        wait_and_throw(exec);

        EXPECT_TRUE(!result, "wrong effect from includes a, c");
    }
};

DEFINE_TEST(test_swap_ranges)
{
    DEFINE_TEST_CONSTRUCTOR(test_swap_ranges, 1.0f, 1.0f)

    template <typename Policy, typename Iterator1, typename Iterator2, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 last2, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);
        TestDataTransfer<UDTKind::eVals, Size> host_vals(*this, n);

        using value_type = typename ::std::iterator_traits<Iterator1>::value_type;
        using reference = typename ::std::iterator_traits<Iterator1>::reference;

        ::std::iota(host_keys.get(), host_keys.get() + n, value_type(0));
        ::std::iota(host_vals.get(), host_vals.get() + n, value_type(n));
        update_data(host_keys, host_vals);

        Iterator2 actual_return = ::std::swap_ranges(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, last1, first2);

        wait_and_throw(exec);

        bool check_return = (actual_return == last2);
        EXPECT_TRUE(check_return, "wrong result of swap_ranges");
        if (check_return)
        {
            ::std::size_t i = 0;

            retrieve_data(host_keys, host_vals);

            auto host_first1 = host_keys.get();
            auto host_first2 = host_vals.get();
            bool check =
                ::std::all_of(host_first2, host_first2 + n, [&i](reference a) { return a == value_type(i++); }) &&
                ::std::all_of(host_first1, host_first1 + n, [&i](reference a) { return a == value_type(i++); });

            EXPECT_TRUE(check, "wrong effect of swap_ranges");
        }
    }
};

DEFINE_TEST(test_reverse_copy)
{
    DEFINE_TEST_CONSTRUCTOR(test_reverse_copy, 1.0f, 1.0f)

    template <typename Policy, typename Iterator1, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first, Iterator1 last, Iterator1 result_first, Iterator1 /* result_last */, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);
        TestDataTransfer<UDTKind::eVals, Size> host_vals(*this, n);

        host_keys.retrieve_data();

        using IteratorValyeType = typename ::std::iterator_traits<Iterator1>::value_type;

        ::std::vector<IteratorValyeType> local_copy(n);
        local_copy.assign(host_keys.get(), host_keys.get() + n);
        ::std::reverse(local_copy.begin(), local_copy.end());

        ::std::reverse_copy(make_new_policy<new_kernel_name<Policy, 0>>(exec), first, last, result_first);
        wait_and_throw(exec);

        host_vals.retrieve_data();
        auto host_first2 = host_vals.get();
        for (int i = 0; i < n; ++i)
            EXPECT_TRUE(local_copy[i] == host_first2[i], "wrong effect from reverse_copy");
    }
};

DEFINE_TEST(test_rotate_copy)
{
    DEFINE_TEST_CONSTRUCTOR(test_rotate_copy, 1.0f, 1.0f)

    template <typename Policy, typename Iterator1, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first, Iterator1 last, Iterator1 result_first, Iterator1 /* result_last */, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);
        TestDataTransfer<UDTKind::eVals, Size> host_vals(*this, n);

        host_keys.retrieve_data();

        using IteratorValyeType = typename ::std::iterator_traits<Iterator1>::value_type;

        ::std::vector<IteratorValyeType> local_copy(n);
        local_copy.assign(host_keys.get(), host_keys.get() + n);
        ::std::rotate(local_copy.begin(), local_copy.begin() + 1, local_copy.end());

        ::std::rotate_copy(make_new_policy<new_kernel_name<Policy, 0>>(exec), first, first + 1, last, result_first);
        wait_and_throw(exec);

        host_vals.retrieve_data();
        for (int i = 0; i < n; ++i)
            EXPECT_TRUE(local_copy[i] == host_vals.get()[i], "wrong effect from rotate_copy");
    }
};

DEFINE_TEST(test_uninitialized_copy)
{
    DEFINE_TEST_CONSTRUCTOR(test_uninitialized_copy, 1.0f, 1.0f)

    template <typename Policy, typename Iterator1, typename Iterator2, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 /* last2 */, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);
        TestDataTransfer<UDTKind::eVals, Size> host_vals(*this, n);

        using IteratorValueType = typename ::std::iterator_traits<Iterator1>::value_type;
        auto value = IteratorValueType(42);

        ::std::fill(host_keys.get(), host_keys.get() + n, value);
        ::std::fill(host_vals.get(), host_vals.get() + n, IteratorValueType{ -1 });
        update_data(host_keys, host_vals);

        ::std::uninitialized_copy(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, last1, first2);
        wait_and_throw(exec);

        host_vals.retrieve_data();
        EXPECT_TRUE(check_values(host_vals.get(), host_vals.get() + n, value), "wrong effect from uninitialized_copy");
    }
};

DEFINE_TEST(test_uninitialized_copy_n)
{
    DEFINE_TEST_CONSTRUCTOR(test_uninitialized_copy_n, 1.0f, 1.0f)

    template <typename Policy, typename Iterator1, typename Iterator2, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 /* last1 */, Iterator2 first2, Iterator2 /* last2 */, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);
        TestDataTransfer<UDTKind::eVals, Size> host_vals(*this, n);

        using IteratorValueType = typename ::std::iterator_traits<Iterator1>::value_type;
        auto value = IteratorValueType(42);

        ::std::fill_n(host_keys.get(), n, value);
        ::std::fill_n(host_vals.get(), n, IteratorValueType{0});
        update_data(host_keys, host_vals);

        ::std::uninitialized_copy_n(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, n, first2);
        wait_and_throw(exec);

        host_vals.retrieve_data();
        EXPECT_TRUE(check_values(host_vals.get(), host_vals.get() + n, value), "wrong effect from uninitialized_copy_n");
    }
};

DEFINE_TEST(test_uninitialized_move)
{
    DEFINE_TEST_CONSTRUCTOR(test_uninitialized_move, 1.0f, 1.0f)

    template <typename Policy, typename Iterator1, typename Iterator2, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 /* last2 */, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);
        TestDataTransfer<UDTKind::eVals, Size> host_vals(*this, n);

        using IteratorValueType = typename ::std::iterator_traits<Iterator1>::value_type;
        auto value = IteratorValueType(42);
        ::std::fill_n(host_keys.get(), n, value);
        ::std::fill_n(host_vals.get(), n, IteratorValueType{ -1 });
        update_data(host_keys, host_vals);

        ::std::uninitialized_move(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, last1, first2);
        wait_and_throw(exec);

        host_vals.retrieve_data();
        EXPECT_TRUE(check_values(host_vals.get(), host_vals.get() + n, value), "wrong effect from uninitialized_move");
    }
};

DEFINE_TEST(test_uninitialized_move_n)
{
    DEFINE_TEST_CONSTRUCTOR(test_uninitialized_move_n, 1.0f, 1.0f)

    template <typename Policy, typename Iterator1, typename Iterator2, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 /* last1 */, Iterator2 first2, Iterator2 /* last2 */, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);
        TestDataTransfer<UDTKind::eVals, Size> host_vals(*this, n);

        using IteratorValueType = typename ::std::iterator_traits<Iterator1>::value_type;
        auto value = IteratorValueType(42);

        ::std::fill_n(host_keys.get(), n, value);
        ::std::fill_n(host_vals.get(), n, IteratorValueType{ -1 });
        update_data(host_keys, host_vals);

        ::std::uninitialized_move_n(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, n, first2);
        wait_and_throw(exec);

        host_vals.retrieve_data();
        EXPECT_TRUE(check_values(host_vals.get(), host_vals.get() + n, value),
                    "wrong effect from uninitialized_move_n");
    }
};

DEFINE_TEST(test_transform_unary)
{
    DEFINE_TEST_CONSTRUCTOR(test_transform_unary, 1.0f, 1.0f)

    template <typename Policy, typename Iterator1, typename Iterator2, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 /* last2 */, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);
        TestDataTransfer<UDTKind::eVals, Size> host_vals(*this, n);

        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;
        auto value = T1(2);

        ::std::fill(host_keys.get(), host_keys.get() + n, value);
        ::std::fill(host_vals.get(), host_vals.get() + n, value + 1);
        update_data(host_keys, host_vals);

        ::std::transform(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1 + n / 2, last1, first2 + n / 2, Flip(7));
        wait_and_throw(exec);

        host_vals.retrieve_data();
        EXPECT_TRUE(check_values(host_vals.get(), host_vals.get() + n / 2, value + 1),
                    "wrong effect from transform_unary (1)");
        EXPECT_TRUE(check_values(host_vals.get() + n / 2, host_vals.get() + n, T1(5)),
                    "wrong effect from transform_unary (2)");
    }
};

DEFINE_TEST(test_transform_binary)
{
    DEFINE_TEST_CONSTRUCTOR(test_transform_binary, 1.0f, 1.0f)

    template <typename Policy, typename Iterator1, typename Iterator2, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 /* last2 */, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);
        TestDataTransfer<UDTKind::eVals, Size> host_vals(*this, n);

        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;
        auto value = T1(3);

        ::std::fill(host_keys.get(), host_keys.get() + n, value);
        host_keys.update_data();

        ::std::transform(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, last1, first1, first2, Plus());
        wait_and_throw(exec);

        host_vals.retrieve_data();
        EXPECT_TRUE(check_values(host_vals.get(), host_vals.get() + n, T1(6)),
                    "wrong effect from transform_binary");
    }
};

DEFINE_TEST(test_replace_copy)
{
    DEFINE_TEST_CONSTRUCTOR(test_replace_copy, 1.0f, 1.0f)

    template <typename Policy, typename Iterator1, typename Iterator2, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 /* last2 */, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);
        TestDataTransfer<UDTKind::eVals, Size> host_vals(*this, n);

        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;
        auto value = T1(5);
        ::std::fill(host_keys.get(), host_keys.get() + n, value);
        host_keys.update_data();

        ::std::replace_copy(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, last1, first2, value, T1(value + 1));
        wait_and_throw(exec);

        host_vals.retrieve_data();
        EXPECT_TRUE(check_values(host_vals.get(), host_vals.get() + n, value + 1),
                    "wrong effect from replace_copy");
    }
};

DEFINE_TEST(test_replace_copy_if)
{
    DEFINE_TEST_CONSTRUCTOR(test_replace_copy_if, 1.0f, 1.0f)

    template <typename Policy, typename Iterator1, typename Iterator2, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 /* last2 */, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);
        TestDataTransfer<UDTKind::eVals, Size> host_vals(*this, n);

        typedef typename ::std::iterator_traits<Iterator1>::value_type T1;
        auto value = T1(6);
        ::std::fill(host_keys.get(), host_keys.get() + n, value);
        host_keys.update_data();

        ::std::replace_copy_if(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, last1, first2,
                             oneapi::dpl::__internal::__equal_value<T1>(value), T1(value + 1));
        wait_and_throw(exec);

        host_vals.retrieve_data();
        EXPECT_TRUE(check_values(host_vals.get(), host_vals.get() + n, value + 1),
                    "wrong effect from replace_copy_if");
    }
};

DEFINE_TEST(test_copy)
{
    DEFINE_TEST_CONSTRUCTOR(test_copy, 1.0f, 1.0f)

    template <typename Policy, typename Iterator1, typename Iterator2, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 /* last2 */, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);
        TestDataTransfer<UDTKind::eVals, Size> host_vals(*this, n);

        using IteratorValueType = typename ::std::iterator_traits<Iterator1>::value_type;
        auto value = IteratorValueType(42);
        ::std::fill(host_keys.get(), host_keys.get() + n, value);
        ::std::fill(host_vals.get(), host_vals.get() + n, IteratorValueType{0});
        update_data(host_keys, host_vals);

        ::std::copy(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, last1, first2);
        wait_and_throw(exec);

        host_vals.retrieve_data();
        EXPECT_TRUE(check_values(host_vals.get(), host_vals.get() + n, value),
                    "wrong effect from copy");
    }
};

DEFINE_TEST(test_copy_n)
{
    DEFINE_TEST_CONSTRUCTOR(test_copy_n, 1.0f, 1.0f)

    template <typename Policy, typename Iterator1, typename Iterator2, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1, Iterator2 first2, Iterator2 /* last2 */, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);
        TestDataTransfer<UDTKind::eVals, Size> host_vals(*this, n);

        using IteratorValueType = typename ::std::iterator_traits<Iterator1>::value_type;
        auto value = IteratorValueType(42);

        ::std::fill(host_keys.get(), host_keys.get() + n, value);
        ::std::fill(host_vals.get(), host_vals.get() + n, IteratorValueType{ 0 });
        update_data(host_keys, host_vals);

        ::std::copy_n(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, n, first2);
        wait_and_throw(exec);

        host_vals.retrieve_data();
        EXPECT_TRUE(check_values(host_vals.get(), host_vals.get() + n, value), "wrong effect from copy_n");
    }
};

DEFINE_TEST(test_move)
{
    DEFINE_TEST_CONSTRUCTOR(test_move, 1.0f, 1.0f)

    template <typename Policy, typename Iterator1, typename Iterator2, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 /* last2 */, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);
        TestDataTransfer<UDTKind::eVals, Size> host_vals(*this, n);

        using IteratorValueType = typename ::std::iterator_traits<Iterator1>::value_type;
        auto value = IteratorValueType(42);

        ::std::fill(host_keys.get(), host_keys.get() + n, value);
        ::std::fill(host_vals.get(), host_vals.get() + n, IteratorValueType{ 0 });
        update_data(host_keys, host_vals);

        ::std::move(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, last1, first2);
        wait_and_throw(exec);

        host_vals.retrieve_data();
        EXPECT_TRUE(check_values(host_vals.get(), host_vals.get() + n, value),
                    "wrong effect from move");
    }
};

DEFINE_TEST(test_adjacent_difference)
{
    DEFINE_TEST_CONSTRUCTOR(test_adjacent_difference, 1.0f, 1.0f)

    template <typename Policy, typename Iterator1, typename Iterator2, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 /* last2 */, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);
        TestDataTransfer<UDTKind::eVals, Size> host_vals(*this, n);

        using Iterator1ValueType = typename ::std::iterator_traits<Iterator1>::value_type;
        using Iterator2ValueType = typename ::std::iterator_traits<Iterator2>::value_type;

        Iterator1ValueType fill_value{1};
        Iterator2ValueType blank_value{0};

        auto __f = [](const Iterator1ValueType& a, const Iterator1ValueType& b) -> Iterator2ValueType { return a + b; };

        // init
        ::std::for_each(host_keys.get(), host_keys.get() + n,
                        [&fill_value](Iterator1ValueType& val) { val = (fill_value++ % 10) + 1; });
        ::std::fill(host_vals.get(), host_vals.get() + n, blank_value);
        update_data(host_keys, host_vals);

        // test with custom functor
        ::std::adjacent_difference(make_new_policy<new_kernel_name<Policy, 0>>(exec), first1, last1, first2, __f);
        wait_and_throw(exec);

        {
            retrieve_data(host_keys, host_vals);

            auto host_first1 = host_keys.get();
            auto host_first2 = host_vals.get();

            bool is_correct = *host_first1 == *host_first2; // for the first element
            for (int i = 1; i < n; ++i)                     // for subsequent elements
                is_correct = is_correct && *(host_first2 + i) == __f(*(host_first1 + i), *(host_first1 + i - 1));

            EXPECT_TRUE(is_correct, "wrong effect from adjacent_difference #1");
        }

        // test with default functor
        ::std::fill(host_vals.get(), host_vals.get() + n, blank_value);
        host_vals.update_data();

        ::std::adjacent_difference(make_new_policy<new_kernel_name<Policy, 1>>(exec), first1, last1, first2);
        wait_and_throw(exec);

        retrieve_data(host_keys, host_vals);

        auto host_first1 = host_keys.get();
        auto host_first2 = host_vals.get();

        bool is_correct = *host_first1 == *host_first2; // for the first element
        for (int i = 1; i < n; ++i)                     // for subsequent elements
            is_correct = is_correct && *(host_first2 + i) == (*(host_first1 + i) - *(host_first1 + i - 1));

        EXPECT_TRUE(is_correct, "wrong effect from adjacent_difference #2");
    }
};

#endif // TEST_DPCPP_BACKEND_PRESENT

#if TEST_DPCPP_BACKEND_PRESENT

using ValueType = ::std::int32_t;

template<>
struct sycl::is_device_copyable<SyclTypeWrapper<ValueType>> : std::true_type {};

template <sycl::usm::alloc alloc_type>
void
test_usm_and_buffer()
{
    // test1buffer
    PRINT_DEBUG("test_for_each");
    test1buffer<alloc_type, test_for_each<ValueType>>();
    PRINT_DEBUG("test_for_each_n");
    test1buffer<alloc_type, test_for_each_n<ValueType>>();
    PRINT_DEBUG("test_replace");
    test1buffer<alloc_type, test_replace<ValueType>>();
    PRINT_DEBUG("test_replace_if");
    test1buffer<alloc_type, test_replace_if<ValueType>>();
    PRINT_DEBUG("test_fill");
    test1buffer<alloc_type, test_fill<ValueType>>();
    PRINT_DEBUG("test_fill_n");
    test1buffer<alloc_type, test_fill_n<ValueType>>();
    PRINT_DEBUG("test_generate");
    test1buffer<alloc_type, test_generate<ValueType>>();
    PRINT_DEBUG("test_generate_n");
    test1buffer<alloc_type, test_generate_n<ValueType>>();
    PRINT_DEBUG("test_reverse");
    test1buffer<alloc_type, test_reverse<ValueType>>();
    PRINT_DEBUG("test_rotate");
    test1buffer<alloc_type, test_rotate<ValueType>>();
    PRINT_DEBUG("test_uninitialized_fill");
    test1buffer<alloc_type, test_uninitialized_fill<ValueType>>();
    PRINT_DEBUG("test_uninitialized_fill_n");
    test1buffer<alloc_type, test_uninitialized_fill_n<ValueType>>();
    PRINT_DEBUG("test_uninitialized_default_construct");
    test1buffer<alloc_type, test_uninitialized_default_construct<SyclTypeWrapper<ValueType>>>();
    PRINT_DEBUG("test_uninitialized_default_construct_n");
    test1buffer<alloc_type, test_uninitialized_default_construct_n<SyclTypeWrapper<ValueType>>>();
    PRINT_DEBUG("test_uninitialized_value_construct");
    test1buffer<alloc_type, test_uninitialized_value_construct<ValueType>>();
    PRINT_DEBUG("test_uninitialized_value_construct_n");
    test1buffer<alloc_type, test_uninitialized_value_construct_n<ValueType>>();
    PRINT_DEBUG("test_destroy");
    test1buffer<alloc_type, test_destroy<SyclTypeWrapper<ValueType>>>();
    PRINT_DEBUG("test_destroy_n");
    test1buffer<alloc_type, test_destroy_n<SyclTypeWrapper<ValueType>>>();
    test1buffer<alloc_type, test_destroy_n<ValueType>>();

    //test2buffers
    PRINT_DEBUG("test_replace_copy");
    test2buffers<alloc_type, test_replace_copy<ValueType>>();
    PRINT_DEBUG("test_replace_copy_if");
    test2buffers<alloc_type, test_replace_copy_if<ValueType>>();
    PRINT_DEBUG("test_transform_unary");
    test2buffers<alloc_type, test_transform_unary<ValueType>>();
    PRINT_DEBUG("test_transform_binary");
    test2buffers<alloc_type, test_transform_binary<ValueType>>();
    PRINT_DEBUG("test_copy");
    test2buffers<alloc_type, test_copy<ValueType>>();
    PRINT_DEBUG("test_copy_n");
    test2buffers<alloc_type, test_copy_n<ValueType>>();
    PRINT_DEBUG("test_move");
    test2buffers<alloc_type, test_move<ValueType>>();
    PRINT_DEBUG("test_adjacent_difference");
    test2buffers<alloc_type, test_adjacent_difference<ValueType>>();
    PRINT_DEBUG("test_swap_ranges");
    test2buffers<alloc_type, test_swap_ranges<ValueType>>();
    PRINT_DEBUG("test_reverse_copy");
    test2buffers<alloc_type, test_reverse_copy<ValueType>>();
    PRINT_DEBUG("test rotate_copy");
    test2buffers<alloc_type, test_rotate_copy<ValueType>>();
    PRINT_DEBUG("test_uninitialized_copy");
    test2buffers<alloc_type, test_uninitialized_copy<ValueType>>();
    PRINT_DEBUG("test_uninitialized_copy_n");
    test2buffers<alloc_type, test_uninitialized_copy_n<ValueType>>();
    PRINT_DEBUG("test_uninitialized_move");
    test2buffers<alloc_type, test_uninitialized_move<ValueType>>();
    PRINT_DEBUG("test_uninitialized_move_n");
    test2buffers<alloc_type, test_uninitialized_move_n<ValueType>>();
    PRINT_DEBUG("test_includes");
    test2buffers<alloc_type, test_includes<ValueType>>();
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
