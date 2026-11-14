// -*- C++ -*-
//===-- equal.pass.cpp ----------------------------------------------------===//
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

struct UserType
{
    size_t key;
    float32_t f;
    std::uint64_t u;
    std::int32_t i;

    bool
    operator()(UserType a, UserType b)
    {
        return a.key < b.key;
    }
    bool
    operator<(UserType a)
    {
        return a.key < key;
    }
    bool
    operator>=(UserType a)
    {
        return a.key <= key;
    }
    bool
    operator<=(UserType a)
    {
        return a.key >= key;
    }
    bool
    operator==(UserType a)
    {
        return a.key == key;
    }
    bool
    operator==(UserType a) const
    {
        return a.key == key;
    }
    bool
    operator!=(UserType a)
    {
        return a.key != key;
    }
    UserType operator!()
    {
        UserType tmp;
        tmp.key = !key;
        return tmp;
    }
    friend ::std::ostream&
    operator<<(::std::ostream& stream, const UserType a)
    {
        stream << a.key;
        return stream;
    }

    UserType() : key(-1), f(0.0f), u(0), i(0) {}
    UserType(size_t Number) : key(Number), f(0.0f), u(0), i(0) {}
    UserType&
    operator=(const UserType& other)
    {
        key = other.key;
        return *this;
    }
    UserType(const UserType& other) : key(other.key), f(other.f), u(other.u), i(other.i) {}
    UserType(UserType&& other) : key(other.key), f(other.f), u(other.u), i(other.i)
    {
        other.key = -1;
        other.f = 0.0f;
        other.u = 0;
        other.i = 0;
    }
};

template <typename T>
struct test_with_4_iters
{
    template <typename ExecutionPolicy, typename Iterator1, typename Iterator2>
    void
    operator()(ExecutionPolicy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 last2, bool is_true_equal)
    {
        auto is_equal = ::std::equal(::std::forward<ExecutionPolicy>(exec), first1, last1, first2, last2);
        EXPECT_TRUE(is_true_equal == is_equal, "result for equal (4 iterators, without predicate) for random-access iterator, bool");
    }

    template <typename ExecutionPolicy, typename Iterator1, typename Iterator2, typename Compare>
    void
    operator()(ExecutionPolicy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 last2, Compare comp, bool is_true_equal)
    {
        auto is_equal = ::std::equal(::std::forward<ExecutionPolicy>(exec), first1, last1, first2, last2, comp);
        EXPECT_TRUE(is_true_equal == is_equal, "result for equal (4 iterators, with predicate) for random-access iterator, bool");
    }
};

template <typename T>
struct test_with_3_iters
{
    template <typename ExecutionPolicy, typename Iterator1, typename Iterator2>
    void
    operator()(ExecutionPolicy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, bool is_true_equal)
    {
        auto is_equal = ::std::equal(::std::forward<ExecutionPolicy>(exec), first1, last1, first2);
        EXPECT_TRUE(is_true_equal == is_equal, "result for equal (3 iterators, without predicate) for random-access iterator, bool");
    }

    template <typename ExecutionPolicy, typename Iterator1, typename Iterator2, typename Compare>
    void
    operator()(ExecutionPolicy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Compare comp, bool is_true_equal)
    {
        auto is_equal = ::std::equal(::std::forward<ExecutionPolicy>(exec), first1, last1, first2, comp);
        EXPECT_TRUE(is_true_equal == is_equal, "result for equal (3 iterators, with predicate) for random-access iterator, bool");
    }
};

template <typename T, typename Compare>
void
test(size_t bits, Compare comp)
{
    const std::vector<std::size_t> sizes = get_find_test_sizes();
    const std::size_t max_size = sizes.back();

    // Sequence of odd values
    Sequence<T> in(max_size, [bits](size_t k) { return T(2 * HashBits(k, bits - 1) ^ 1); });
    Sequence<T> inCopy(in);

    for (size_t n : sizes)
    {
        const bool ex_res_true = true;
        invoke_on_all_policies_with_host_cap<0>{n}(test_with_3_iters<T>(), in.begin(), in.begin() + n, inCopy.begin(), comp, ex_res_true);
        invoke_on_all_policies_with_host_cap<1>{n}(test_with_3_iters<T>(), in.cbegin(), in.cbegin() + n,
                                                   inCopy.cbegin(), ex_res_true);
        invoke_on_all_policies_with_host_cap<2>{n}(test_with_4_iters<T>(), in.begin(), in.begin() + n, inCopy.begin(), inCopy.begin() + n, comp, ex_res_true);

        // testing bool !equal()
        T original = inCopy[0];
        inCopy[0] = !original;
        const bool ex_res_false = false || n == 0; //In case of the both sequences are empty, result is true.
        invoke_on_all_policies_with_host_cap<3>{n}(test_with_4_iters<T>(), in.begin(), in.begin() + n, inCopy.begin(), inCopy.begin() + n, ex_res_false);
        invoke_on_all_policies_with_host_cap<4>{n}(test_with_4_iters<T>(), in.cbegin(), in.cbegin() + n, inCopy.cbegin(), inCopy.cbegin() + n, comp, ex_res_false);
        invoke_on_all_policies_with_host_cap<5>{n}(test_with_3_iters<T>(), in.cbegin(), in.cbegin() + n, inCopy.cbegin(), ex_res_false);
        inCopy[0] = original;

        // A single difference in the last element
        if (n > 0)
        {
            original = inCopy[n - 1];
            inCopy[n - 1] = !original;
            invoke_on_all_policies_with_host_cap<8>{n}(test_with_4_iters<T>(), in.begin(), in.begin() + n,
                                                       inCopy.begin(), inCopy.begin() + n, false);
            inCopy[n - 1] = original;
        }
    }
    // check different sized sequences
    invoke_on_all_policies<6>()(test_with_4_iters<T>(), in.begin(), in.begin() + max_size - 1, inCopy.begin(), inCopy.begin() + max_size, false);
    invoke_on_all_policies<7>()(test_with_4_iters<T>(), in.cbegin(), in.cbegin() + max_size, inCopy.cbegin(), inCopy.cbegin() + max_size - 1, comp, false);
}

template <typename T>
struct test_non_const
{
    template <typename Policy, typename FirstIterator, typename SecondInterator>
    void
    operator()(Policy&& exec, FirstIterator first_iter, SecondInterator second_iter)
    {
        equal(::std::forward<Policy>(exec), first_iter, first_iter, second_iter, second_iter, non_const(::std::equal_to<T>()));
    }
};

int
main()
{
    test<std::int32_t>(8 * sizeof(std::int32_t),   [](const std::int32_t& a, const std::int32_t& b)     { return a == b; });
    test<std::uint16_t>(8 * sizeof(std::uint16_t), [](const std::uint16_t& a, const std::uint16_t& b)   { return a == b; });
    test<float64_t>(53,                  [](const float64_t& a, const float64_t& b) { return a == b; });
    test<bool>(1,                        [](const bool& a, const bool& b)           { return a == b; });

    // Avoiding UserType test for dpcpp backend, because UserType is not a device copyable type, with a custom copy
    // assignment operator which is not the same as a bitwise copy. Therefore, a sycl::buffer with the value type
    // UserType is not valid.
#if !TEST_DPCPP_BACKEND_PRESENT
    test<UserType>(256,                  [](const UserType& a, const UserType& b)   { return a == b; });
#endif

    test_algo_basic_double<std::int32_t>(run_for_rnd_fw<test_non_const<std::int32_t>>());

    return done();
}
