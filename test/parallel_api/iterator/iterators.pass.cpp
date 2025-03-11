// -*- C++ -*-
//===-- iterators.pass.cpp ------------------------------------------------===//
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

#include _PSTL_TEST_HEADER(iterator)
#include _PSTL_TEST_HEADER(execution)
#include _PSTL_TEST_HEADER(algorithm)

#include "support/utils.h"

#include <vector>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <limits>
#include <type_traits>
#include <forward_list>

using namespace TestUtils;

//common checks of a random access iterator functionality which is needed for oneDPL algos
template <typename RandomIt>
void
test_random_iterator_skip_default_ctor_check(const RandomIt& it)
{
    // check that RandomIt has all necessary publicly accessible member types
    {
        [[maybe_unused]] auto t1 = typename RandomIt::difference_type{};
        [[maybe_unused]] auto t2 = typename RandomIt::value_type{};
        [[maybe_unused]] auto t3 = typename RandomIt::pointer{};
        [[maybe_unused]] typename RandomIt::reference ref = *it;
        [[maybe_unused]] auto t4 = typename RandomIt::iterator_category{};
    }

    EXPECT_TRUE(  it == it,      "== returned false negative");
    EXPECT_TRUE(!(it == it + 1), "== returned false positive");
    EXPECT_TRUE(  it != it + 1,  "!= returned false negative");
    EXPECT_TRUE(!(it != it),     "!= returned false positive");

    EXPECT_TRUE(*it == *it, "wrong result with operator*");

    RandomIt it1 = it;
    EXPECT_TRUE(it1 == it, "iterator is not copy constructible");
    RandomIt it2 = RandomIt(it);
    EXPECT_TRUE(it2 == it, "iterator is not move constructible");

    ++it1;
    EXPECT_TRUE(it1 == it + 1, "wrong result with prefix operator++");

    using ::std::swap;
    swap(it1, it2);
    EXPECT_TRUE((it1 == it) && (it2 == it + 1), "iterator is not swappable");

    it2 = it;
    EXPECT_TRUE(it2 == it, "iterator is not copy assignable");

    ++it2;
    it2 = RandomIt(it);
    EXPECT_TRUE(it2 == it, "iterator is not move assignable");

    it1 = it;
    EXPECT_TRUE((it1++ == it) && (it1 == it + 1), "wrong result with postfix operator++");

    it1 = it + 1;
    EXPECT_TRUE(--it1 == it, "wrong result with prefix operator--");

    it1 = it + 1;
    EXPECT_TRUE((it1-- == it + 1) && (it1 == it), "wrong result with postfix operator--");

    it1 += 1;
    EXPECT_TRUE(it1 == it + 1, "wrong result with operator+=");

    it1 -= 1;
    EXPECT_TRUE(it1 == it, "wrong result with operator-=");

    EXPECT_TRUE(1 + it == it + 1, "n + iterator != iterator + n");

    EXPECT_TRUE((it + 1) - 1 == it, "wrong result with operator-(difference_type)");

    EXPECT_TRUE((it + 1) - it == 1, "wrong result with iterator subtraction");

    // There is a bug in clang when we pass the same arguments in the function
    if(it[1]!=*(it + 1)){
        ::std::cout<<"wrong result with operator[]"<<::std::endl;
        exit(1);
    }

    EXPECT_TRUE(it < it + 1, "operator< returned false negative");
    EXPECT_TRUE(!(it < it),  "operator< returned false positive");

    EXPECT_TRUE(it + 1 > it, "operator> returned false negative");
    EXPECT_TRUE(!(it > it),  "operator> returned false positive");

    EXPECT_TRUE(it <= it + 1,    "operator<= returned false negative");
    EXPECT_TRUE(it <= it,        "operator<= returned false negative");
    EXPECT_TRUE(!(it + 1 <= it), "operator<= returned false positive");

    EXPECT_TRUE(1 + it >= it,    "operator>= returned false negative");
    EXPECT_TRUE(    it >= it,    "operator>= returned false negative");
    EXPECT_TRUE(!(it >= it + 1), "operator>= returned false positive");
}

template <typename RandomIt>
void
test_random_iterator(const RandomIt& it)
{
    static_assert(std::is_default_constructible_v<RandomIt>, "iterator is not default constructible");
    test_random_iterator_skip_default_ctor_check(it);
}

struct test_counting_iterator {
    template <typename T, typename IntType>
    void operator()( ::std::vector<T>& in, IntType begin, IntType end, const T& value) {
        EXPECT_TRUE((0 <= begin) && (begin <= end) && (end <= IntType(in.size())),
        "incorrect test_counting_iterator 'begin' and/or 'end' argument values");

        //test that counting_iterator is default constructible
        oneapi::dpl::counting_iterator<IntType> b;

        b = oneapi::dpl::counting_iterator<IntType>(begin);
        auto e = oneapi::dpl::counting_iterator<IntType>(end);

        //checks in using
        ::std::for_each(b, e, [&in, &value](IntType i) { in[i] = value; });

        auto res = ::std::all_of(in.begin(), in.begin() + begin, [&value](const T& a) {return a!=value;});
        EXPECT_TRUE(res, "wrong result with counting_iterator in vector's begin portion");

        res = ::std::all_of(in.begin() + begin, in.begin() + end, [&value](const T& a) {return a==value;});
        EXPECT_TRUE(res, "wrong result with counting_iterator in vector's main portion");

        res = ::std::all_of(in.begin() + end, in.end(), [&value](const T& a) {return a!=value;});
        EXPECT_TRUE(res, "wrong result with counting_iterator in vector's end portion");

        //explicit checks of the counting iterator specific
        EXPECT_TRUE(*(b + 1) == begin+1, "wrong result with operator+ for an iterator");
        EXPECT_TRUE(*(b+=1) == begin+1, "wrong result with operator+= for an iterator");

        test_random_iterator(b);
    }
};

struct sort_fun{
    template<typename T1, typename T2>
    bool operator()(T1 a1, T2 a2) const {
        return ::std::get<0>(a1) < ::std::get<0>(a2);
    }
};

template <typename InputIterator>
void test_explicit_move(InputIterator i, InputIterator j) {
    using value_type = typename ::std::iterator_traits<InputIterator>::value_type;
    value_type t(::std::move(*i));
    *i = ::std::move(*j);
    *j = ::std::move(t);
}

struct test_zip_iterator {
    template <typename T1, typename T2>
    void operator()(::std::vector<T1>& in1, ::std::vector<T2>& in2) {
        //runtime call to check default constructor and increase code coverage
        oneapi::dpl::zip_iterator<decltype(in1.begin()), decltype(in2.begin())> b;
        //runtime call to check constructor with variadic arguments and increase code coverage
        oneapi::dpl::zip_iterator<decltype(in1.begin()), decltype(in2.begin())> c(in1.begin(), in2.begin());

        //runtime call to check copy assignable operator and increase code coverage
        b = oneapi::dpl::make_zip_iterator(in1.begin(), in2.begin());
        auto e = oneapi::dpl::make_zip_iterator(in1.end(), in2.end());

        EXPECT_TRUE( (b+1) != e, "size of input sequence insufficient for test" );

        //simple check for-loop.
        {
        ::std::for_each(b, e, [](const ::std::tuple<T1&, T2&>& a) { ::std::get<0>(a) = 1, ::std::get<1>(a) = 1;});
        auto res = ::std::all_of(b, e, [](const ::std::tuple<T1&, T2&>& a) {return ::std::get<0>(a) == 1 && ::std::get<1>(a) == 1;});
        EXPECT_TRUE(res, "wrong result sequence assignment to (1,1) with zip_iterator iterator");
        //all_of check for iterator which is constructed passing variadic arguments
        res = ::std::all_of(c, e, [](const ::std::tuple<T1&, T2&>& a) {return ::std::get<0>(a) == 1 && ::std::get<1>(a) == 1;});
        EXPECT_TRUE(res, "wrong result of all_of algorithm with zip_iterator iterator which is constructed passing variadic arguments");
        }

        //check swapping de-referenced iterators (required by sort algorithm)
        {
        using ::std::swap;
        auto t = ::std::make_tuple(T1(3), T2(2));
        *b = t;
        t = *(b + 1);
        EXPECT_TRUE(::std::get<0>(t) == 1 && ::std::get<1>(t) == 1, "wrong result of assignment from zip_iterator");
        swap(*b, *(b + 1));
        EXPECT_TRUE(::std::get<0>(*b) == 1 && ::std::get<1>(*b) == 1, "wrong result swapping zip-iterator");
        EXPECT_TRUE(::std::get<0>(*(b + 1)) == 3 && ::std::get<1>(*(b + 1)) == 2, "wrong result swapping zip-iterator");
        // Test leaves sequence un-sorted.
        }

        //sort sequences by first stream.
        {
        // sanity check if sequence is un-sorted.
        auto res = ::std::is_sorted(b, e, sort_fun());
        EXPECT_TRUE(!res, "input sequence to be sorted is already sorted! Test might lead to false positives.");
        ::std::sort(oneapi::dpl::make_zip_iterator(in1.begin(), in2.begin()),
                  oneapi::dpl::make_zip_iterator(in1.end(), in2.end()),
                  sort_fun());
        res = ::std::is_sorted(b, e, sort_fun());
        EXPECT_TRUE(res, "wrong result sorting sequence using zip-iterator");
            // TODO: Add simple check: comparison with sort_fun().
        }
        test_explicit_move(b, b+1);
        auto iter_base = b.base();
        EXPECT_TRUE(::std::get<0>(iter_base) == in1.begin(), "wrong result from base (get<0>)");
        EXPECT_TRUE(::std::get<1>(iter_base) == in2.begin(), "wrong result from base (get<1>)");

        test_random_iterator(b);
    }
};

template <typename VecIt1, typename VecIt2>
void test_transform_effect(VecIt1 first1, VecIt1 last1, VecIt2 first2) {
    auto triple = [](typename ::std::iterator_traits<VecIt1>::value_type const& val) {
        return typename ::std::iterator_traits<VecIt2>::value_type (3 * val);
    };

    ::std::copy(
        oneapi::dpl::make_transform_iterator(first1, triple),
        oneapi::dpl::make_transform_iterator(last1,  triple),
        first2
    );

    for (typename ::std::iterator_traits<VecIt1>::difference_type i = 0; i < last1 - first1; ++i)
        if ( first2[i] != (typename ::std::iterator_traits<VecIt2>::value_type) triple(first1[i]) ) {
            ::std::cout << "wrong effect with transform iterator" << ::std::endl;
            exit(1);
        }
}

//We need this functor to run fill algorithm with transform iterators. Operator() should return lvalue reference.
struct ref_transform_functor {
    template<typename T>
    T& operator()(T& x) const {
        return x += 1;
    }
};

//We need this functor to pass operator* test for transform iterator
struct transform_functor {
    template<typename T>
    T operator()(T& x) const {
        return x + 1;
    }
};

struct test_transform_iterator {
    template <typename T1, typename T2>
    void operator()(::std::vector<T1>& in1, ::std::vector<T2>& in2) {
        ::std::iota(in1.begin(), in1.end(), T1(0));

        test_transform_effect(in1.begin(),  in1.end(),  in2.begin());
        test_transform_effect(in1.cbegin(), in1.cend(), in2.begin());

        transform_functor new_functor;
        ref_transform_functor ref_functor;
        //check default constructibility of transform_iterator with default constructible components
        [[maybe_unused]] oneapi::dpl::transform_iterator<T1*, transform_functor> _it0;
        oneapi::dpl::transform_iterator<typename ::std::vector<T1>::iterator, transform_functor> _it1(in1.begin());
        oneapi::dpl::transform_iterator<typename ::std::vector<T1>::iterator, transform_functor> _it2(in1.begin(), new_functor);

        ::std::forward_list<int> f_list{1, 2, 3, 4, 5, 6};
        auto list_it1 = oneapi::dpl::make_transform_iterator(f_list.begin(), ref_functor);
        auto list_it2 = oneapi::dpl::make_transform_iterator(f_list.end(), ref_functor);
        ::std::fill(list_it1, list_it2, 7);
        EXPECT_TRUE(::std::all_of(f_list.begin(), f_list.end(), [](int x){ return x == 7; }),
            "wrong result from fill with forward_iterator wrapped with transform_iterator");

        auto test_lambda = [](T2& x){ return x + 1; };
        auto new_transform_iterator = oneapi::dpl::make_transform_iterator(in2.begin(), test_lambda);
        EXPECT_TRUE(_it1.base() == in1.begin(), "wrong result from transform_iterator::base");
        static_assert(::std::is_same_v<decltype(new_transform_iterator.functor()), decltype(test_lambda)>,
            "wrong result from transform_iterator::functor");
        test_random_iterator(_it2);
    }
};

struct test_permutation_iterator
{
    template <typename T1, typename T2>
    void
    operator()(::std::vector<T1>& in1, ::std::vector<T2>& in2)
    {
        T1 iota_max = ::std::numeric_limits<T1>::max() < in1.size() ? ::std::numeric_limits<T1>::max() : in1.size();
        ::std::iota(in1.begin(), in1.begin() + iota_max, T1(0));
        ::std::reverse_copy(in1.begin(), in1.begin() + iota_max, in2.begin());

        oneapi::dpl::permutation_iterator<typename ::std::vector<T1>::iterator, typename ::std::vector<T2>::iterator> perm_begin;
        perm_begin = oneapi::dpl::make_permutation_iterator(in1.begin(), in2.begin());
        auto perm_end = oneapi::dpl::make_permutation_iterator(in1.begin(), in2.begin()) + iota_max;

        ::std::vector<T1> result(iota_max);
        ::std::copy(perm_begin, perm_end, result.begin());

        EXPECT_TRUE(::std::is_sorted(result.begin(), result.end(), ::std::greater<T1>()),
                    "wrong result from copy with permutation_iterator");

        oneapi::dpl::permutation_iterator<typename ::std::vector<T1>::iterator, typename ::std::vector<T2>::iterator> perm_it1(in1.begin(), in2.begin());
        oneapi::dpl::permutation_iterator<typename ::std::vector<T1>::iterator, typename ::std::vector<T2>::iterator> perm_it2(in1.begin(), in2.begin() + in2.size()-1);
        EXPECT_TRUE(perm_it1 == perm_begin, "wrong result from permutation_iterator(base_iterator, index_iterator)");
        EXPECT_TRUE(perm_it2 == perm_begin + in2.size()-1, "wrong result from permutation_iterator(base_iterator, index_iterator)");
        EXPECT_TRUE(perm_it1.base() == in1.begin(), "wrong result from permutation_iterator::base_iterator");
        EXPECT_TRUE(perm_it1.map() == in2.begin(), "wrong result from permutation_iterator::index_iterator");

        test_random_iterator(perm_begin);

        auto n = in1.size();
        auto stateful_lambda = [n] (auto i) { return n - i - 1;};
        auto perm_it_fun_rev = oneapi::dpl::make_permutation_iterator(in1.begin(), stateful_lambda, 1);
        EXPECT_TRUE(*++perm_it_fun_rev == *(in1.end()-3), "wrong result from permutation_iterator(base_iterator, functor)");
        //stateful lambdas won't be default constructible, therefore this permutation iterator will not be either
        test_random_iterator_skip_default_ctor_check(perm_it_fun_rev);

        auto stateless_lambda = [] (auto i) { return i;};
        auto perm_it_stateless_fun = oneapi::dpl::make_permutation_iterator(in1.begin(), stateless_lambda, 1);
        EXPECT_TRUE(*++perm_it_stateless_fun == in1[2], "wrong result from permutation_iterator(base_iterator, stateless_lambda)");

        //stateless lambdas may be default constructible based on the c++ standard, permutation iterator should match
        if constexpr(std::is_default_constructible_v<decltype(stateless_lambda)>)
            test_random_iterator(perm_it_stateless_fun);
        else
            test_random_iterator_skip_default_ctor_check(perm_it_stateless_fun);


        ::std::vector<T1> res(n);
        perm_it_fun_rev -= 2;
        ::std::copy(perm_it_fun_rev, perm_it_fun_rev + n, res.begin());

        EXPECT_EQ_N(res.begin(), oneapi::dpl::make_reverse_iterator(in1.end()), n, "wrong result from permutation_iterator(base_iterator, functor)");
    }
};

struct test_discard_iterator
{
    template <typename T1, typename T2>
    void
    operator()(::std::vector<T1>& in1, ::std::vector<T2>& in2)
    {
        ::std::iota(in1.begin(), in1.end(), T1(0));

        oneapi::dpl::discard_iterator dis_it;
        oneapi::dpl::discard_iterator dis_it2(in1.size());

        EXPECT_TRUE(dis_it + in1.size() == dis_it2, "wrong result from discard_iterator");

        test_random_iterator(dis_it);
    }
};

template <typename T, typename IntType>
void test_iterator_by_type(IntType n) {

    const IntType beg = 0;
    const IntType end = n;

    ::std::vector<T> in(n, T(0));
    ::std::vector<IntType> in2(n, IntType(0));

    test_counting_iterator()(in, beg,     end,     /*value*/ T(-1));
    test_counting_iterator()(in, beg+123, end-321, /*value*/ T(42));
    test_random_iterator(oneapi::dpl::counting_iterator<IntType>(beg));

    test_zip_iterator()(in, in2);
    test_transform_iterator()(in, in2);
    test_permutation_iterator()(in, in2);
    test_discard_iterator()(in, in2);
}

int main() {
    const auto n1 = 1000;
    const auto n2 = 100000;

    test_iterator_by_type<std::int16_t, std::int16_t>(n1);
    test_iterator_by_type<std::int16_t, std::int64_t>(n2);

    test_iterator_by_type<double, std::int16_t>(n1);
    test_iterator_by_type<double, std::int64_t>(n2);

    return done();
}
