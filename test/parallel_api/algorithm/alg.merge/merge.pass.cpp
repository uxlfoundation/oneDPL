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
#include <string>
#include <string_view>
#include <vector>
#include <iterator>
#include <algorithm>
#include <numeric>
#include <tuple>
#include <utility>

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
        const auto res = merge(std::forward<Policy>(exec), first1, last1, first2, last2, out_first);
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
        using T = typename std::iterator_traits<std::reverse_iterator<InputIterator1>>::value_type;
        const auto res = merge(std::forward<Policy>(exec), first1, last1, first2, last2, out_first, std::greater<T>());

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
        const auto res = merge(std::forward<Policy>(exec), first1, last1, first2, last2, out_first, comp);
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
        using T = typename std::iterator_traits<std::reverse_iterator<InputIterator1>>::value_type;
        const auto res = merge(std::forward<Policy>(exec), first1, last1, first2, last2, out_first, std::greater<T>());

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
    test_merge_by_type<std::int32_t>([](std::int32_t v) { return (v % 2 == 0 ? v : -v) * 3; },
                                     [](std::int32_t v) { return v * 2; }, start_size, max_size, fstep);
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
        merge(std::forward<Policy>(exec), input_iter, input_iter, input_iter, input_iter, out_iter, non_const(std::less<T>()));
    }
};

// A (key, value) pair compared by the key only: the order of the values reveals the stability.
// Unlike std::tuple, it is trivially assignable, which makes the parallel merge take a different code path.
struct KeyValue
{
    std::int32_t key;
    std::int32_t value;

    friend bool
    operator==(const KeyValue& x, const KeyValue& y)
    {
        return x.key == y.key && x.value == y.value;
    }
};

using KeyValueTuple = std::tuple<std::int32_t, std::int32_t>;

std::int32_t
get_key(const KeyValue& x)
{
    return x.key;
}

std::int32_t
get_key(const KeyValueTuple& x)
{
    return std::get<0>(x);
}

enum class KeysOverlap
{
    interleaved, // the keys of {1} and {2} cover the same range
    touching,    // {1} is ordered before {2}, the last key of {1} is equal to the first key of {2}
    reversed     // {2} is ordered before {1}
};

template <std::size_t CallNumber, typename T>
void
test_stability_and_order_impl(std::size_t a_size, std::size_t b_size, KeysOverlap overlap)
{
    // The keys grow from 0 to 99 in both sequences, forming runs of equal keys of different lengths.
    // The keys of {1} are shifted to get the requested overlap.
    // The values are unique, so stability fully determines the result.
    constexpr std::size_t n_keys = 100;
    const std::int32_t last_key_a = std::int32_t((a_size - 1) * n_keys / a_size);
    const std::int32_t shift_a = overlap == KeysOverlap::touching   ? -last_key_a
                                 : overlap == KeysOverlap::reversed ? std::int32_t(n_keys)
                                                                    : 0;
    std::vector<T> a(a_size), b(b_size);
    std::vector<T> merged(a.size() + b.size());
    std::vector<T> expected(a.size() + b.size());

    for (std::size_t i = 0; i < a_size; ++i)
        a[i] = T{shift_a + std::int32_t(i * n_keys / a_size), std::int32_t(i)};
    for (std::size_t i = 0; i < b_size; ++i)
        b[i] = T{std::int32_t(i * n_keys / b_size), std::int32_t(a_size + i)};

    auto comp = [](const T& x, const T& y) { return get_key(x) < get_key(y); }; // less by key
    std::merge(a.begin(), a.end(), b.begin(), b.end(), expected.begin(), comp);

    auto test_merge_stable = [&](auto&& exec, auto first1, auto last1, auto first2, auto last2, auto out_first)
    {
        // Reverse iterators break the order of the keys
        if constexpr (!is_reverse_v<decltype(first1)>)
        {
            std::merge(std::forward<decltype(exec)>(exec), first1, last1, first2, last2, out_first, comp);
            EXPECT_EQ_N(expected.begin(), out_first, expected.size(), "wrong merge result with equal keys");
        }
    };
    invoke_on_all_policies<CallNumber>()(test_merge_stable, a.begin(), a.end(), b.cbegin(), b.cend(), merged.begin());
}

void
test_stability_and_order()
{
    for (KeysOverlap overlap : {KeysOverlap::interleaved, KeysOverlap::touching, KeysOverlap::reversed})
    {
        test_stability_and_order_impl<100, KeyValue>(61, 77, overlap);
        test_stability_and_order_impl<200, KeyValue>(5711, 20429, overlap);
        test_stability_and_order_impl<400, KeyValueTuple>(10000, 8000, overlap);
    }
}

#if !TEST_DPCPP_BACKEND_PRESENT
template <typename Policy>
void
test_merge_move_iterators_impl(Policy&& exec, int n1, int n2)
{
    std::vector<std::string> in1(n1), in2(n2), out(n1 + n2);
    std::vector<int> keys1(n1), keys2(n2), keys_expected(n1 + n2), keys_out(n1 + n2);

    // 1. Generate alternating blocks for keys1 and keys2:
    //    keys1: [0:49, 100:149, 200:249, ...]
    //    keys2: [50:99, 150:199, 250:299, ...]
    //    Once one of the sequences is exhausted, the other one continues with the remaining keys
    constexpr int block = 50;
    for (int i = 0; i < n1; ++i)
        keys1[i] = i + std::min(i / block * block, n2);
    for (int i = 0; i < n2; ++i)
        keys2[i] = i + std::min((i / block + 1) * block, n1);

    // 2. Together the keys cover [0, n1 + n2) without duplicates, hence the merged keys are just a sequence
    std::iota(keys_expected.begin(), keys_expected.end(), 0);

    // 3. Encode the same keys into strings
    constexpr std::string_view base_str("i_am_long_enough_to_avoid_SSO"); // SSO-ed string may not be emptied when moved
    const std::size_t key_width = std::to_string(n1 + n2).size();
    auto key_to_string = [base_str, key_width](int key) {
        const std::string key_str = std::to_string(key);
        const std::string padding(key_width - key_str.size(), '0');
        return std::string(base_str) + padding + key_str; // it always has the same length, regardless of the key value
    };
    std::transform(keys1.begin(), keys1.end(), in1.begin(), key_to_string);
    std::transform(keys2.begin(), keys2.end(), in2.begin(), key_to_string);

    // 4. Merge these strings using the same order as the keys
    auto res = std::merge(std::forward<Policy>(exec),
                          std::make_move_iterator(in1.begin()), std::make_move_iterator(in1.end()),
                          std::make_move_iterator(in2.begin()), std::make_move_iterator(in2.end()),
                          out.begin());

    // 5. Extract the merged keys from the merged strings, and compare them with the expected keys
    auto string_to_key = [base_str](const std::string& s) {
        // check the size since substr throws for an empty (moved-from) string in case of corrupted output
        return s.size() > base_str.size() ? std::stoi(s.substr(base_str.size())) : -1;
    };
    std::transform(out.begin(), out.end(), keys_out.begin(), string_to_key);
    EXPECT_TRUE(res == out.end(), "wrong return result from merge with move iterators");
    EXPECT_EQ_RANGES(keys_expected, keys_out, "wrong result from merge with move iterators");
}

void
test_merge_move_iterators()
{
    const int sizes[][2] = {{3000, 3000}, {1000, 20000}, {50000, 7000}};
    for (const auto& s : sizes)
    {
        test_merge_move_iterators_impl(oneapi::dpl::execution::seq, s[0], s[1]);
        test_merge_move_iterators_impl(oneapi::dpl::execution::unseq, s[0], s[1]);
        test_merge_move_iterators_impl(oneapi::dpl::execution::par, s[0], s[1]);
        test_merge_move_iterators_impl(oneapi::dpl::execution::par_unseq, s[0], s[1]);
    }
}
#endif // !TEST_DPCPP_BACKEND_PRESENT

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

    // Data is moved instead of copied: checking for possible data races
#if !TEST_DPCPP_BACKEND_PRESENT
    test_merge_move_iterators();
#endif

    test_stability_and_order();

    return done();
}
