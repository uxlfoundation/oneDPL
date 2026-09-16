// -*- C++ -*-
//===-- find_first_of.pass.cpp --------------------------------------------===//
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
struct test_find_first_of
{
    template <typename ExecutionPolicy, typename Iterator1, typename Iterator2>
    void
    operator()(ExecutionPolicy&& exec, Iterator1 b, Iterator1 e, Iterator2 bsub, Iterator2 esub)
    {
        using namespace std;
        Iterator1 expected = find_first_of(b, e, bsub, esub);
        Iterator1 actual = find_first_of(std::forward<ExecutionPolicy>(exec), b, e, bsub, esub);
        EXPECT_EQ(expected, actual, "wrong return result from find_first_of");
    }
};

template <typename T>
struct test_find_first_of_predicate
{
    template <typename ExecutionPolicy, typename Iterator1, typename Iterator2, typename Predicate>
    void
    operator()(ExecutionPolicy&& exec, Iterator1 b, Iterator1 e, Iterator2 bsub, Iterator2 esub, Predicate pred)
    {
        using namespace std;
        Iterator1 expected = find_first_of(b, e, bsub, esub, pred);
        Iterator1 actual = find_first_of(std::forward<ExecutionPolicy>(exec), b, e, bsub, esub, pred);
        EXPECT_EQ(expected, actual, "wrong return result from find_first_of with a predicate");
    }
};

template <typename T, typename Predicate>
void
test(Predicate pred)
{

    const ::std::size_t max_n1 = 1000;
    const ::std::size_t max_n2 = (max_n1 * 10) / 8;
    Sequence<T> in1(max_n1, [](::std::size_t) { return T(1); });
    Sequence<T> in2(max_n2, [](::std::size_t) { return T(0); });
    for (::std::size_t n1 = 0; n1 <= max_n1; n1 = n1 <= 16 ? n1 + 1 : size_t(3.1415 * n1))
    {
        ::std::size_t sub_n[] = {0, 1, n1 / 3, n1, (n1 * 10) / 8};
        for (const auto n2 : sub_n)
        {
            invoke_on_all_policies<0>()(test_find_first_of<T>(), in1.begin(), in1.begin() + n1, in2.begin(),
                                        in2.begin() + n2);
            invoke_on_all_policies<1>()(test_find_first_of_predicate<T>(), in1.begin(), in1.begin() + n1, in2.begin(),
                                        in2.begin() + n2, pred);

            in2[n2 / 2] = T(1);
#if !TEST_DPCPP_BACKEND_PRESENT
            invoke_on_all_policies<2>()(test_find_first_of<T>(), in1.cbegin(), in1.cbegin() + n1, in2.data(),
                                        in2.data() + n2);
            invoke_on_all_policies<3>()(test_find_first_of_predicate<T>(), in1.cbegin(), in1.cbegin() + n1, in2.data(),
                                        in2.data() + n2, pred);
#else
#if !ONEDPL_FPGA_DEVICE
            invoke_on_all_policies<2>()(test_find_first_of<T>(), in1.cbegin(), in1.cbegin() + n1, in2.begin(),
                                        in2.begin() + n2);
            invoke_on_all_policies<3>()(test_find_first_of_predicate<T>(), in1.cbegin(), in1.cbegin() + n1, in2.begin(),
                                        in2.begin() + n2, pred);
#endif
#endif
            if (n2 >= 3)
            {
                in2[2 * n2 / 3] = T(1);
                invoke_on_all_policies<4>()(test_find_first_of<T>(), in1.cbegin(), in1.cbegin() + n1, in2.begin(),
                                            in2.begin() + n2);
                invoke_on_all_policies<5>()(test_find_first_of_predicate<T>(), in1.cbegin(), in1.cbegin() + n1,
                                            in2.begin(), in2.begin() + n2, pred);
                in2[2 * n2 / 3] = T(0);
            }
            in2[n2 / 2] = T(0);
        }
    }
    invoke_on_all_policies<6>()(test_find_first_of<T>(), in1.begin(), in1.begin() + max_n1 / 10, in1.begin(),
                                in1.begin() + max_n1 / 10);
    invoke_on_all_policies<7>()(test_find_first_of_predicate<T>(), in1.begin(), in1.begin() + max_n1 / 10, in1.begin(),
                                in1.begin() + max_n1 / 10, pred);
}

// The data of test() above pins none of the answers: all the elements of its first range are equal, so
// whenever a match exists it is at the very first of them, and all of its predicates are symmetric. An
// implementation which compares the first element of the first range only, or which passes the two
// elements to the predicate in the opposite order, answers exactly like a correct one on every case
// test() generates. The cases below drop both properties: the first range holds distinct values, its
// matching elements are away from its front, and one of the predicates tells the two argument orders
// apart.
//
// They also compare the position of the answer instead of the iterator, because EXPECT_EQ cannot print
// an iterator: a failure of the two functors above reports only "expected (unable to log value)", which
// says nothing about which element was returned.
template <typename T>
struct test_find_first_of_position
{
    template <typename ExecutionPolicy, typename Iterator1, typename Iterator2>
    void
    operator()(ExecutionPolicy&& exec, Iterator1 b, Iterator1 e, Iterator2 bsub, Iterator2 esub)
    {
        using namespace std;
        // distance, not a subtraction: invoke_on_all_policies instantiates this with the forward and
        // bidirectional iterator wrappers of the test suite as well
        const auto expected = distance(b, find_first_of(b, e, bsub, esub));
        const auto actual = distance(b, find_first_of(std::forward<ExecutionPolicy>(exec), b, e, bsub, esub));
        EXPECT_EQ(expected, actual, "wrong position from find_first_of");
    }
};

template <typename T>
struct test_find_first_of_position_predicate
{
    template <typename ExecutionPolicy, typename Iterator1, typename Iterator2, typename Predicate>
    void
    operator()(ExecutionPolicy&& exec, Iterator1 b, Iterator1 e, Iterator2 bsub, Iterator2 esub, Predicate pred)
    {
        using namespace std;
        const auto expected = distance(b, find_first_of(b, e, bsub, esub, pred));
        const auto actual = distance(b, find_first_of(std::forward<ExecutionPolicy>(exec), b, e, bsub, esub, pred));
        EXPECT_EQ(expected, actual, "wrong position from find_first_of with a predicate");
    }
};

// A vectorized find_first_of walks whichever of the two ranges is the shorter one in its inner loop, so
// it has one such loop per order of the two lengths. This is the order with the first range being the
// shorter one, the one the SIMD brick serves with its per element scan of the second range.
template <typename T>
void
test_match_away_from_the_front()
{
    // is_successor(x, y) and is_successor(y, x) hold for different elements of the first range
    auto is_successor = [](const T x, const T y) { return x == T(y + 1); };

    const ::std::size_t sizes[] = {2, 3, 7, 16, 41, 130};
    for (const auto n1 : sizes)
    {
        // distinct values, and T(0), which the second range is filled with, is not one of them
        Sequence<T> in1(n1, [](::std::size_t i) { return T(10 + i); });

        // The second range is the longer one. Its two matching elements are planted in the order
        // opposite to the one their counterparts have in the first range, so the answer is the earliest
        // match in the first range rather than the first match met while walking the second one.
        Sequence<T> in2(n1 + 3, [](::std::size_t) { return T(0); });
        in2[0] = T(10 + n1 - 1);
        in2[1] = T(10 + n1 / 2);
        invoke_on_all_policies<8>()(test_find_first_of_position<T>(), in1.begin(), in1.end(), in2.begin(), in2.end());

        // The same lengths with a predicate which is not symmetric: is_successor(*it1, *it2) holds for
        // the last element of the first range, while the swapped is_successor(*it2, *it1) would hold for
        // the third element from its end.
        in2[0] = T(0);
        in2[1] = T(10 + n1 - 2);
        invoke_on_all_policies<9>()(test_find_first_of_position_predicate<T>(), in1.begin(), in1.end(), in2.begin(),
                                    in2.end(), is_successor);
    }
}

// The other order of the two lengths, the first range being the longer one, is served by the branch which
// walks the *second* range in its outer loop. Keeping the earliest match over all the elements of that
// range is a requirement of its own, separate from the argument order and from the per element binding of
// the branch above: an implementation which returns the first match it meets answers 6 for n1 = 7, where
// in1 = {10, ..., 16}, in2 = {16, 13, 0} and find_first_of answers 3.
template <typename T>
void
test_match_away_from_the_front_long_first_range()
{
    const ::std::size_t sizes[] = {2, 3, 7, 16, 41, 130};
    for (const auto n1 : sizes)
    {
        Sequence<T> in1(n1, [](::std::size_t i) { return T(10 + i); });

        // The same two matches as above, planted in the same reversed order, with the second range being
        // the shorter one this time.
        Sequence<T> in2(n1 / 2 + 1, [](::std::size_t) { return T(0); });
        in2[0] = T(10 + n1 - 1);
        if (in2.size() > 1)
            in2[1] = T(10 + n1 / 2);
        invoke_on_all_policies<10>()(test_find_first_of_position<T>(), in1.begin(), in1.end(), in2.begin(), in2.end());
    }
}

template <typename T>
struct test_non_const
{
    template <typename Policy, typename FirstIterator, typename SecondInterator>
    void
    operator()(Policy&& exec, FirstIterator first_iter, SecondInterator second_iter)
    {
        find_first_of(std::forward<Policy>(exec), first_iter, first_iter, second_iter, second_iter, non_const(std::equal_to<T>()));
    }
};

int
main()
{
    test<std::int32_t>(::std::equal_to<std::int32_t>());
#if !ONEDPL_FPGA_DEVICE
    test<std::uint16_t>(::std::not_equal_to<std::uint16_t>());
#endif
    test<float64_t>([](const float64_t x, const float64_t y) { return x * x == y * y; });

    test_match_away_from_the_front<std::int32_t>();
    test_match_away_from_the_front_long_first_range<std::int32_t>();

    test_algo_basic_double<std::int32_t>(run_for_rnd_fw<test_non_const<std::int32_t>>());

    return done();
}
