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

#ifndef _SET_COMMON_H
#define _SET_COMMON_H

#include "support/test_config.h"

#include _PSTL_TEST_HEADER(execution)
#include _PSTL_TEST_HEADER(algorithm)

#include "support/utils.h"

#include <cmath>
#include <chrono>

using namespace TestUtils;

template <typename T>
struct Num
{
    T val;

    Num() : val{} {}
    Num(const T& v) : val(v) {}

    //for "includes" checks
    template <typename T1>
    bool
    operator<(const Num<T1>& v1) const
    {
        return val < v1.val;
    }

    //The types Type1 and Type2 must be such that an object of type InputIt can be dereferenced and then implicitly converted to both of them
    template <typename T1>
    operator Num<T1>() const
    {
        return Num<T1>((T1)val);
    }

    friend bool
    operator==(const Num& v1, const Num& v2)
    {
        return v1.val == v2.val;
    }
};

template<typename InputIterator1, typename InputIterator2>
auto
init(InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, InputIterator2 last2) ->
::std::pair<Sequence<typename ::std::iterator_traits<InputIterator1>::value_type>,
            Sequence<typename ::std::iterator_traits<InputIterator1>::value_type>>
{
    using T1 = typename ::std::iterator_traits<InputIterator1>::value_type;

    auto n1 = ::std::distance(first1, last1);
    auto n2 = ::std::distance(first2, last2);
    auto n = n1 + n2;
    Sequence<T1> expect(n);
    Sequence<T1> out(n);
    return ::std::make_pair(expect,out);
}

template <typename Type>
struct test_set_union
{
    template <typename Policy, typename InputIterator1, typename InputIterator2, typename Compare>
    ::std::enable_if_t<!TestUtils::is_reverse_v<InputIterator1>>
    operator()(Policy&& exec, InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, InputIterator2 last2,
               Compare comp)
    {

        auto sequences = init(first1, last1, first2, last2);
        auto expect = sequences.first;
        auto out = sequences.second;
        auto expect_res = ::std::set_union(first1, last1, first2, last2, expect.begin(), comp);
        auto res = ::std::set_union(exec, first1, last1, first2, last2, out.begin(), comp);

        EXPECT_TRUE(expect_res - expect.begin() == res - out.begin(), "wrong result for set_union");
        EXPECT_EQ_N(expect.begin(), out.begin(), ::std::distance(out.begin(), res), "wrong set_union effect");
    }

    template <typename Policy, typename InputIterator1, typename InputIterator2>
    ::std::enable_if_t<!TestUtils::is_reverse_v<InputIterator1>>
    operator()(Policy&& exec, InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, InputIterator2 last2)
    {
        auto sequences = init(first1, last1, first2, last2);
        auto expect = sequences.first;
        auto out = sequences.second;
        auto expect_res = ::std::set_union(first1, last1, first2, last2, expect.begin());
        auto res = ::std::set_union(exec, first1, last1, first2, last2, out.begin());

        EXPECT_TRUE(expect_res - expect.begin() == res - out.begin(), "wrong result for set_union without comparator");
        EXPECT_EQ_N(expect.begin(), out.begin(), ::std::distance(out.begin(), res), "wrong set_union effect without comparator");
    }

    template <typename Policy, typename InputIterator1, typename InputIterator2>
    ::std::enable_if_t<TestUtils::is_reverse_v<InputIterator1>>
    operator()(Policy&&, InputIterator1, InputIterator1, InputIterator2, InputIterator2)
    {
    }

    template <typename Policy, typename InputIterator1, typename InputIterator2, typename Compare>
    ::std::enable_if_t<TestUtils::is_reverse_v<InputIterator1>>
    operator()(Policy&&, InputIterator1, InputIterator1, InputIterator2, InputIterator2, Compare)
    {
    }
};

// Compare the first of a tuple using the supplied comparator
template <typename _Comp>
struct comp_select_first
{
    _Comp comp;
    comp_select_first(_Comp __comp) : comp(__comp) {}
    template <typename _T1, typename _T2>
    bool
    operator()(_T1&& t1, _T2&& t2) const
    {
        return comp(std::get<0>(std::forward<_T1>(t1)), std::get<0>(std::forward<_T2>(t2)));
    }
};

template <typename Type>
struct test_set_intersection
{
    template <typename Policy, typename InputIterator1, typename InputIterator2, typename Compare>
    ::std::enable_if_t<!TestUtils::is_reverse_v<InputIterator1>>
    operator()(Policy&& exec, InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, InputIterator2 last2,
               Compare comp)
    {
        auto sequences = init(first1, last1, first2, last2);
        auto expect = sequences.first;
        auto out = sequences.second;
        auto expect_res = ::std::set_intersection(first1, last1, first2, last2, expect.begin(), comp);
        auto res = ::std::set_intersection(create_new_policy_idx<0>(std::forward<Policy>(exec)), first1, last1, first2,
                                           last2, out.begin(), comp);

        EXPECT_TRUE(expect_res - expect.begin() == res - out.begin(), "wrong result for set_intersection");
        EXPECT_EQ_N(expect.begin(), out.begin(), ::std::distance(out.begin(), res), "wrong set_intersection effect");

        if constexpr (TestUtils::is_base_of_iterator_category<std::random_access_iterator_tag,
                                                              InputIterator1>::value &&
                      TestUtils::is_base_of_iterator_category<std::random_access_iterator_tag, InputIterator2>::value)
        {
            // Check that set_intersection always copies from the first list to result.
            // Will fail to compile if the second range is used to copy to the output.
            // Comparator is designed to only compare the first element of a zip iterator.

            auto zip_first1 = oneapi::dpl::make_zip_iterator(first1, oneapi::dpl::counting_iterator<int>(0));
            auto zip_last1 = oneapi::dpl::make_zip_iterator(
                last1, oneapi::dpl::counting_iterator<int>(std::distance(first1, last1)));

            // Second value should be ignored and discarded in range 2 because the result should be copied from range 1
            auto zip_first2 = oneapi::dpl::make_zip_iterator(first2, oneapi::dpl::discard_iterator());
            auto zip_last2 = oneapi::dpl::make_zip_iterator(last2, oneapi::dpl::discard_iterator());

            Sequence<int> expect_ints(std::distance(first1, last1) + std::distance(first2, last2));
            Sequence<int> out_ints(std::distance(first1, last1) + std::distance(first2, last2));

            auto zip_expect = oneapi::dpl::make_zip_iterator(sequences.first.begin(), expect_ints.begin());
            auto zip_out = oneapi::dpl::make_zip_iterator(sequences.second.begin(), out_ints.begin());

            auto zip_expect_res = std::set_intersection(zip_first1, zip_last1, zip_first2, zip_last2, zip_expect,
                                                        comp_select_first(comp));
            auto zip_res = std::set_intersection(create_new_policy_idx<1>(std::forward<Policy>(exec)), zip_first1,
                                                 zip_last1, zip_first2, zip_last2, zip_out, comp_select_first(comp));
            EXPECT_TRUE(zip_expect_res - zip_expect == zip_res - zip_out, "wrong result for zipped set_intersection");
            EXPECT_EQ_N(zip_expect, zip_out, std::distance(zip_out, zip_res), "wrong zipped set_intersection effect");
        }
    }

    template <typename Policy, typename InputIterator1, typename InputIterator2>
    ::std::enable_if_t<!TestUtils::is_reverse_v<InputIterator1>>
    operator()(Policy&& exec, InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, InputIterator2 last2)
    {
        auto sequences = init(first1, last1, first2, last2);
        auto expect = sequences.first;
        auto out = sequences.second;
        auto expect_res = ::std::set_intersection(first1, last1, first2, last2, expect.begin());
        auto res = ::std::set_intersection(exec, first1, last1, first2, last2, out.begin());

        EXPECT_TRUE(expect_res - expect.begin() == res - out.begin(), "wrong result for set_intersection without comparator");
        EXPECT_EQ_N(expect.begin(), out.begin(), ::std::distance(out.begin(), res), "wrong set_intersection effect without comparator");
    }

    template <typename Policy, typename InputIterator1, typename InputIterator2>
    ::std::enable_if_t<TestUtils::is_reverse_v<InputIterator1>>
    operator()(Policy&&, InputIterator1, InputIterator1, InputIterator2, InputIterator2)
    {
    }

    template <typename Policy, typename InputIterator1, typename InputIterator2, typename Compare>
    ::std::enable_if_t<TestUtils::is_reverse_v<InputIterator1>>
    operator()(Policy&&, InputIterator1, InputIterator1, InputIterator2, InputIterator2, Compare)
    {
    }
};

template <typename Type>
struct test_set_difference
{
    template <typename Policy, typename InputIterator1, typename InputIterator2, typename Compare>
    ::std::enable_if_t<!TestUtils::is_reverse_v<InputIterator1>>
    operator()(Policy&& exec, InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, InputIterator2 last2,
               Compare comp)
    {
        auto sequences = init(first1, last1, first2, last2);
        auto expect = sequences.first;
        auto out = sequences.second;
        auto expect_res = ::std::set_difference(first1, last1, first2, last2, expect.begin(), comp);
        auto res = ::std::set_difference(exec, first1, last1, first2, last2, out.begin(), comp);

        EXPECT_TRUE(expect_res - expect.begin() == res - out.begin(), "wrong result for set_difference");
        EXPECT_EQ_N(expect.begin(), out.begin(), ::std::distance(out.begin(), res), "wrong set_difference effect");
    }

    template <typename Policy, typename InputIterator1, typename InputIterator2>
    ::std::enable_if_t<!TestUtils::is_reverse_v<InputIterator1>>
    operator()(Policy&& exec, InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, InputIterator2 last2)
    {
        auto sequences = init(first1, last1, first2, last2);
        auto expect = sequences.first;
        auto out = sequences.second;
        auto expect_res = ::std::set_difference(first1, last1, first2, last2, expect.begin());
        auto res = ::std::set_difference(exec, first1, last1, first2, last2, out.begin());

        EXPECT_TRUE(expect_res - expect.begin() == res - out.begin(),
                    "wrong result for set_difference without comparator");
        EXPECT_EQ_N(expect.begin(), out.begin(), ::std::distance(out.begin(), res),
                    "wrong set_difference effect without comparator");
    }

    template <typename Policy, typename InputIterator1, typename InputIterator2>
    ::std::enable_if_t<TestUtils::is_reverse_v<InputIterator1>>
    operator()(Policy&&, InputIterator1, InputIterator1, InputIterator2, InputIterator2)
    {
    }

    template <typename Policy, typename InputIterator1, typename InputIterator2, typename Compare>
    ::std::enable_if_t<TestUtils::is_reverse_v<InputIterator1>>
    operator()(Policy&&, InputIterator1, InputIterator1, InputIterator2, InputIterator2, Compare)
    {
    }
};

template <typename Type>
struct test_set_symmetric_difference
{
    template <typename Policy, typename InputIterator1, typename InputIterator2, typename Compare>
    ::std::enable_if_t<!TestUtils::is_reverse_v<InputIterator1>>
    operator()(Policy&& exec, InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, InputIterator2 last2,
               Compare comp)
    {
        auto sequences = init(first1, last1, first2, last2);
        auto expect = sequences.first;
        auto out = sequences.second;
        auto expect_res = ::std::set_symmetric_difference(first1, last1, first2, last2, expect.begin(), comp);
        auto res = ::std::set_symmetric_difference(exec, first1, last1, first2, last2, out.begin(), comp);

        EXPECT_TRUE(expect_res - expect.begin() == res - out.begin(), "wrong result for set_symmetric_difference");
        EXPECT_EQ_N(expect.begin(), out.begin(), ::std::distance(out.begin(), res),
                    "wrong set_symmetric_difference effect");
    }

    template <typename Policy, typename InputIterator1, typename InputIterator2>
    ::std::enable_if_t<!TestUtils::is_reverse_v<InputIterator1>>
    operator()(Policy&& exec, InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, InputIterator2 last2)
    {
        auto sequences = init(first1, last1, first2, last2);
        auto expect = sequences.first;
        auto out = sequences.second;
        auto expect_res = ::std::set_symmetric_difference(first1, last1, first2, last2, expect.begin());
        auto res = ::std::set_symmetric_difference(exec, first1, last1, first2, last2, out.begin());

        EXPECT_TRUE(expect_res - expect.begin() == res - out.begin(),
                    "wrong result for set_symmetric_difference without comparator");
        EXPECT_EQ_N(expect.begin(), out.begin(), ::std::distance(out.begin(), res),
                    "wrong set_symmetric_difference effect without comparator");
    }

    template <typename Policy, typename InputIterator1, typename InputIterator2>
    ::std::enable_if_t<TestUtils::is_reverse_v<InputIterator1>>
    operator()(Policy&&, InputIterator1, InputIterator1, InputIterator2, InputIterator2)
    {
    }

    template <typename Policy, typename InputIterator1, typename InputIterator2, typename Compare>
    ::std::enable_if_t<TestUtils::is_reverse_v<InputIterator1>>
    operator()(Policy&&, InputIterator1, InputIterator1, InputIterator2, InputIterator2, Compare)
    {
    }
};

template <template <typename T> typename TestType, typename T1, typename T2, typename Compare>
void
test_set(Compare compare, bool comp_flag)
{
    const ::std::size_t n_max = 100000;

    // The rand()%(2*n+1) encourages generation of some duplicates.
    ::std::srand(4200);

    for (::std::size_t n = 0; n < n_max; n = n <= 16 ? n + 1 : size_t(3.1415 * n))
    {
        for (::std::size_t m = 0; m < n_max; m = m <= 16 ? m + 1 : size_t(2.71828 * m))
        {
            //prepare the input ranges trying to get some amount of overlap
            Sequence<T1> in1(n, [](std::size_t k) { return rand() % (std::max(std::size_t{3},k>>4)); });
            Sequence<T2> in2(m, [m](std::size_t k) { return ((m % 2) * rand() + rand()) % (std::max(std::size_t{3},k>>4)); });

            ::std::sort(in1.begin(), in1.end(), compare);
            ::std::sort(in2.begin(), in2.end(), compare);

            if (comp_flag)
                invoke_on_all_policies<0>()(TestType<T1>(), in1.begin(), in1.end(), in2.cbegin(), in2.cend(), compare);
            else
                invoke_on_all_policies<1>()(TestType<T1>(), in1.begin(), in1.end(), in2.cbegin(), in2.cend());
        }
    }
}

template <template <typename T> typename TestType>
void
run_test_set()
{
    using data_t =
#if !ONEDPL_FPGA_DEVICE
        float64_t;
#else
        std::int32_t;
#endif

    test_set<TestType, data_t, data_t>(oneapi::dpl::__internal::__pstl_less(), false);
#if !ONEDPL_FPGA_DEVICE
    test_set<TestType, data_t, data_t>(oneapi::dpl::__internal::__pstl_less(), true);
#endif // !ONEDPL_FPGA_DEVICE

#if !TEST_DPCPP_BACKEND_PRESENT
    test_set<TestType, Num<std::int64_t>, Num<std::int32_t>>(
        [](const Num<std::int64_t>& x, const Num<std::int32_t>& y) { return x < y; }, true);

    test_set<TestType, MemoryChecker, MemoryChecker>(
        [](const MemoryChecker& val1, const MemoryChecker& val2) -> bool { return val1.value() < val2.value(); }, true);
    EXPECT_TRUE(MemoryChecker::alive_objects() == 0,
                "wrong effect from set algorithms: number of ctor and dtor calls is not equal");
#endif // !TEST_DPCPP_BACKEND_PRESENT
}

#endif // _SET_COMMON_H
