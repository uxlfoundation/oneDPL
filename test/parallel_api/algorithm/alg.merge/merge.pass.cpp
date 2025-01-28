// -*- C++ -*-
//===-- merge.pass.cpp ----------------------------------------------------===//
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

#include <functional>

using namespace TestUtils;

template <typename Type>
struct test_merge
{
    template <typename Policy, typename InputIterator1, typename InputIterator2, typename OutputIterator>
    void
    operator()(Policy&& exec, InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, InputIterator2 last2,
               OutputIterator out_first, OutputIterator out_last)
    {
        using namespace std;
        const auto res = merge(exec, first1, last1, first2, last2, out_first);
        EXPECT_TRUE(res == out_last, "wrong return result from merge");
        EXPECT_TRUE(is_sorted(out_first, res), "wrong result from merge");
    }

    // for reverse iterators
    template <typename Policy, typename InputIterator1, typename InputIterator2, typename OutputIterator>
    void
    operator()(Policy&& exec, std::reverse_iterator<InputIterator1> first1, std::reverse_iterator<InputIterator1> last1,
               std::reverse_iterator<InputIterator2> first2, std::reverse_iterator<InputIterator2> last2,
               std::reverse_iterator<OutputIterator> out_first, std::reverse_iterator<OutputIterator> out_last)
    {
        using namespace std;
        typedef typename std::iterator_traits<std::reverse_iterator<InputIterator1>>::value_type T;
        const auto res = merge(exec, first1, last1, first2, last2, out_first, std::greater<T>());

        EXPECT_TRUE(res == out_last, "wrong return result from merge with predicate");
        EXPECT_TRUE(is_sorted(out_first, res, std::greater<T>()), "wrong result from merge with predicate");
        EXPECT_TRUE(includes(out_first, res, first1, last1, std::greater<T>()),
                    "first sequence is not a part of result");
        EXPECT_TRUE(includes(out_first, res, first2, last2, std::greater<T>()),
                    "second sequence is not a part of result");
    }
};

template <typename Type>
struct test_merge_compare
{
    template <typename Policy, typename InputIterator1, typename InputIterator2, typename OutputIterator,
    typename Compare>
    void
    operator()(Policy&& exec, InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, InputIterator2 last2,
               OutputIterator out_first, OutputIterator out_last, Compare comp)
    {
        using namespace std;
        const auto res = merge(exec, first1, last1, first2, last2, out_first, comp);
        EXPECT_TRUE(res == out_last, "wrong return result from merge with predicate");
        EXPECT_TRUE(is_sorted(out_first, res, comp), "wrong result from merge with predicate");
        EXPECT_TRUE(includes(out_first, res, first1, last1, comp), "first sequence is not a part of result");
        EXPECT_TRUE(includes(out_first, res, first2, last2, comp), "second sequence is not a part of result");
    }

    // for reverse iterators
    template <typename Policy, typename InputIterator1, typename InputIterator2, typename OutputIterator,
    typename Compare>
    void
    operator()(Policy&& exec, std::reverse_iterator<InputIterator1> first1, std::reverse_iterator<InputIterator1> last1,
               std::reverse_iterator<InputIterator2> first2, std::reverse_iterator<InputIterator2> last2,
               std::reverse_iterator<OutputIterator> out_first, std::reverse_iterator<OutputIterator> out_last,
               Compare /* comp */)
    {
        using namespace std;
        typedef typename std::iterator_traits<std::reverse_iterator<InputIterator1>>::value_type T;
        const auto res = merge(exec, first1, last1, first2, last2, out_first, std::greater<T>());

        EXPECT_TRUE(res == out_last, "wrong return result from merge with predicate");
        EXPECT_TRUE(is_sorted(out_first, res, std::greater<T>()), "wrong result from merge with predicate");
        EXPECT_TRUE(includes(out_first, res, first1, last1, std::greater<T>()),
                    "first sequence is not a part of result");
        EXPECT_TRUE(includes(out_first, res, first2, last2, std::greater<T>()),
                    "second sequence is not a part of result");
    }
};

template <typename T, typename Generator1, typename Generator2, typename FStep>
void
test_merge_by_type(Generator1 generator1, Generator2 generator2, size_t start_size, size_t max_size, FStep fstep)
{
    using namespace std;
    Sequence<T> in1(max_size, generator1);
    Sequence<T> in2(max_size / 2, generator2);
    Sequence<T> out(in1.size() + in2.size());
    std::sort(in1.begin(), in1.end());
    std::sort(in2.begin(), in2.end());

    for (size_t size = start_size; size <= max_size; size = fstep(size)) {
#if !TEST_DPCPP_BACKEND_PRESENT
        invoke_on_all_policies<0>()(test_merge<T>(),  in1.cbegin(), in1.cbegin() + size,  in2.data(),
                                    in2.data() + size / 2, out.begin(), out.begin() + 1.5 * size);
        invoke_on_all_policies<1>()(test_merge_compare<T>(), in1.cbegin(), in1.cbegin() + size, in2.data(),
                                    in2.data() + size / 2, out.begin(), out.begin() + 1.5 * size, std::less<T>());
#endif

        // Currently test harness doesn't execute the testcase for inputs with more than 1000 elements for const iterators to optimize execution time,
        // but merge's parallel version cut off size is equal to 2000. By using a non-const iterator the testcase can be executed for sizes > 1000 and therefore executing
        // the parallel version of merge.
        invoke_on_all_policies<2>()(test_merge<T>(),  in1.begin(), in1.begin() + size,  in2.cbegin(),
                                    in2.cbegin() + size / 2, out.begin(), out.begin() + 1.5 * size);
        invoke_on_all_policies<3>()(test_merge_compare<T>(), in1.begin(), in1.begin() + size, in2.cbegin(),
                                    in2.cbegin() + size / 2, out.begin(), out.begin() + 1.5 * size, std::less<T>());

#if !TEST_DPCPP_BACKEND_PRESENT
        invoke_on_all_policies<4>()(test_merge<T>(), in1.data(), in1.data() + size, in2.cbegin(),
                                    in2.cbegin() + size / 2, out.begin(), out.begin() + 3 * size / 2);
        invoke_on_all_policies<5>()(test_merge_compare<T>(), in1.data(), in1.data() + size, in2.cbegin(),
                                    in2.cbegin() + size / 2, out.begin(), out.begin() + 3 * size / 2, std::less<T>());
#endif
    }
}

template <typename FStep>
void
test_merge_by_type(size_t start_size, size_t max_size, FStep fstep)
{
    test_merge_by_type<std::int32_t>([](size_t v) { return (v % 2 == 0 ? v : -v) * 3; }, [](size_t v) { return v * 2; }, start_size, max_size, fstep);
#if !ONEDPL_FPGA_DEVICE
    test_merge_by_type<float64_t>([](size_t v) { return float64_t(v); }, [](size_t v) { return float64_t(v - 100); }, start_size, max_size, fstep);
#endif

#if !TEST_DPCPP_BACKEND_PRESENT
    // Wrapper has atomic increment in ctor. It's not allowed in kernel
    test_merge_by_type<Wrapper<std::int16_t>>([](size_t v) { return Wrapper<std::int16_t>(v % 100); },
                                              [](size_t v) { return Wrapper<std::int16_t>(v % 10); },
                                              start_size, max_size, fstep);
#endif
}

template <typename T>
struct test_non_const
{
    template <typename Policy, typename InputIterator, typename OutputIterator>
    void
    operator()(Policy&& exec, InputIterator input_iter, OutputIterator out_iter)
    {
        merge(exec, input_iter, input_iter, input_iter, input_iter, out_iter, non_const(std::less<T>()));
    }
};

struct test_merge_tuple
{
    template <typename Policy, typename InputIterator1, typename InputIterator2, typename OutputIterator,
    typename Compare, typename Checker>
    void
    operator()(Policy&& exec, InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, InputIterator2 last2,
               OutputIterator out_first, Compare comp, Checker check)
    {
        std::merge(exec, first1, last1, first2, last2, out_first, comp);
        check();
    }
};

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    const size_t start_size_small = 2;
#else
    const size_t start_size_small = 0;
#endif
    const size_t max_size_small = 100000;
    auto fstep_small = [](std::size_t size){ return size <= 16 ? size + 1 : size_t(3.1415 * size);};
    test_merge_by_type(start_size_small, max_size_small, fstep_small);

    // Large data sizes
#if TEST_DPCPP_BACKEND_PRESENT
    const size_t start_size_large = 4'000'000;
    const size_t max_size_large = 8'000'000;
    auto fstep_large = [](std::size_t size){ return size + 2'000'000; };
    test_merge_by_type(start_size_large, max_size_large, fstep_large);
#endif

#if !TEST_DPCPP_BACKEND_PRESENT
    test_algo_basic_double<std::int32_t>(run_for_rnd_fw<test_non_const<std::int32_t>>());
#endif


    using T = std::tuple<std::int32_t, std::int32_t>; //a pair (key, value)
    std::vector<T> a = { {1, 2}, {1, 2}, {1,2}, {1,2}, {1, 2}, {1, 2} };
    std::vector<T> b = { {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1} };
    std::vector<T> merged(a.size() + b.size());

    auto comp = [](auto a, auto b) { return std::get<0>(b) < std::get<0>(a); }; //greater by key

    invoke_on_all_policies<100>()(test_merge_tuple(), a.begin(), a.end(), b.cbegin(), b.cend(), merged.begin(), comp,
        [&]()
        {
            std::int32_t sum1 = 0; //a sum of the first a.size() values, should be 2*a.size()
            std::int32_t sum2 = 0; //a sum of the second b.size() values, should be 1*b.size()
            for(std::int32_t i = 0; i < a.size(); ++i)
                sum1 += std::get<1>(merged[i]);
            for(std::int32_t i = 0; i < b.size(); ++i)
                sum2 += std::get<1>(merged[a.size() + i]);

            EXPECT_TRUE(sum1 == 2*a.size(), "wrong merge return with tuple");
            EXPECT_TRUE(sum2 == 1*b.size(), "wrong merge return with tuple");
        });

    return done();
}
