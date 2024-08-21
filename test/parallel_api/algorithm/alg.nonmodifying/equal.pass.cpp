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
    constexpr ::std::size_t max_size = 100000;

    // Sequence of odd values
    Sequence<T> in(max_size, [bits](size_t k) { return T(2 * HashBits(k, bits - 1) ^ 1); });
    Sequence<T> inCopy(in);

    for (size_t n = 1; n <= max_size; n = n <= 16 ? n + 1 : size_t(3.1415 * n))
    {
        invoke_on_all_policies<0>()(test_with_3_iters<T>(), in.begin(), in.begin() + n, inCopy.begin(), comp, true);
        invoke_on_all_policies<1>()(test_with_3_iters<T>(), in.cbegin(), in.cbegin() + n, inCopy.cbegin(), true);
        invoke_on_all_policies<2>()(test_with_4_iters<T>(), in.begin(), in.begin() + n, inCopy.begin(), inCopy.begin() + n, comp, true);

        // testing bool !equal()
        T original = inCopy[0];
        inCopy[0] = !original;
        invoke_on_all_policies<3>()(test_with_4_iters<T>(), in.begin(), in.begin() + n, inCopy.begin(), inCopy.begin() + n, false);
        invoke_on_all_policies<4>()(test_with_4_iters<T>(), in.cbegin(), in.cbegin() + n, inCopy.cbegin(), inCopy.cbegin() + n, comp, false);
        invoke_on_all_policies<5>()(test_with_3_iters<T>(), in.cbegin(), in.cbegin() + n, inCopy.cbegin(), false);
        inCopy[0] = original;
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
