// -*- C++ -*-
//===-- replace_copy.pass.cpp ---------------------------------------------===//
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

// Tests for replace_copy and replace_copy_if

#include "support/test_config.h"

#include _PSTL_TEST_HEADER(execution)
#include _PSTL_TEST_HEADER(algorithm)

#include "support/utils.h"

#if  !defined(_PSTL_TEST_REPLACE_COPY) && !defined(_PSTL_TEST_REPLACE_COPY_IF)
#define _PSTL_TEST_REPLACE_COPY
#define _PSTL_TEST_REPLACE_COPY_IF
#endif

using namespace TestUtils;

template <typename t>
struct test_replace_copy
{
    template <typename Policy, typename InputIterator, typename OutputIterator, typename OutputIterator2, typename Size,
              typename Predicate, typename T>
    void
    operator()(Policy&& exec, InputIterator first, InputIterator last, OutputIterator out_first,
               OutputIterator out_last, OutputIterator2 expected_first, OutputIterator2 /* expected_last */, Size n,
               Predicate /* pred */, const T& old_value, const T& new_value, T trash)
    {
        // Cleaning
        ::std::fill_n(expected_first, n, trash);
        ::std::fill_n(out_first, n, trash);
        // Run replace_copy
        ::std::replace_copy(first, last, expected_first, old_value, new_value);
        auto k = std::replace_copy(std::forward<Policy>(exec), first, last, out_first, old_value, new_value);
        EXPECT_EQ_N(expected_first, out_first, n, "wrong replace_copy effect");
        EXPECT_TRUE(out_last == k, "wrong return value from replace_copy");
    }
};

template <typename t>
struct test_replace_copy_if
{
    template <typename Policy, typename InputIterator, typename OutputIterator, typename OutputIterator2, typename Size,
              typename Predicate, typename T>
    void
    operator()(Policy&& exec, InputIterator first, InputIterator last, OutputIterator out_first,
               OutputIterator out_last, OutputIterator2 expected_first, OutputIterator2 /* expected_last */, Size n,
               Predicate pred, const T& /* pld_value */, const T& new_value, T trash)
    {
        // Cleaning
        ::std::fill_n(expected_first, n, trash);
        ::std::fill_n(out_first, n, trash);
        // Run replace_copy_if
        replace_copy_if(first, last, expected_first, pred, new_value);
        auto k = replace_copy_if(std::forward<Policy>(exec), first, last, out_first, pred, new_value);
        EXPECT_EQ_N(expected_first, out_first, n, "wrong replace_copy_if effect");
        EXPECT_TRUE(out_last == k, "wrong return value from replace_copy_if");
    }
};

template <typename T, typename Convert, typename Predicate>
void
test(T trash, const T& old_value, const T& new_value, Predicate pred, Convert convert)
{
    // Try sequences of various lengths.
    for (size_t n : TestUtils::get_pattern_for_test_sizes())
    {
        Sequence<T> in(n, [&](size_t k) -> T { return convert(n ^ k); });
        Sequence<T> out(n, [=](size_t) { return trash; });
        Sequence<T> expected(n, [=](size_t) { return trash; });

#ifdef _PSTL_TEST_REPLACE_COPY
        invoke_on_all_policies<0>()(test_replace_copy<T>(), in.begin(), in.end(), out.begin(), out.end(),
                                    expected.begin(), expected.end(), out.size(), pred, old_value, new_value, trash);
        invoke_on_all_policies<1>()(test_replace_copy<T>(), in.cbegin(), in.cend(), out.begin(), out.end(),
                                    expected.begin(), expected.end(), out.size(), pred, old_value, new_value, trash);
#endif
#ifdef _PSTL_TEST_REPLACE_COPY_IF
        invoke_on_all_policies<2>()(test_replace_copy_if<T>(), in.begin(), in.end(), out.begin(), out.end(),
                                    expected.begin(), expected.end(), out.size(), pred, old_value, new_value, trash);
        invoke_on_all_policies<3>()(test_replace_copy_if<T>(), in.cbegin(), in.cend(), out.begin(), out.end(),
                                    expected.begin(), expected.end(), out.size(), pred, old_value, new_value, trash);
#endif
    }
}

template <typename T>
struct test_non_const
{
    template <typename Policy, typename InputIterator, typename OutputInterator>
    void
    operator()(Policy&& exec, InputIterator input_iter, OutputInterator out_iter)
    {
        auto is_even = TestUtils::IsEven<float64_t>{};
        replace_copy_if(std::forward<Policy>(exec), input_iter, input_iter, out_iter, non_const(is_even), T(0));
    }
};

struct not_implicitly_convertible
{
    explicit not_implicitly_convertible(int v) {}
};

template <typename It, typename DestIt, typename Void = void>
struct is_replace_copy_well_formed : std::false_type {};

template <typename It, typename DestIt>
struct is_replace_copy_well_formed<It, DestIt,
                                   std::void_t<decltype(oneapi::dpl::replace_copy(oneapi::dpl::execution::seq,
                                                                                  std::declval<It>(),
                                                                                  std::declval<It>(),
                                                                                  std::declval<DestIt>(),
                                                                                  {2},
                                                                                  {3}))>> : std::true_type {};

template <typename It, typename DestIt, typename Void = void>
struct is_replace_copy_if_well_formed : std::false_type {};

template <typename It, typename DestIt>
struct is_replace_copy_if_well_formed<It, DestIt,
                                      std::void_t<decltype(oneapi::dpl::replace_copy_if(oneapi::dpl::execution::seq,
                                                                                        std::declval<It>(),
                                                                                        std::declval<It>(),
                                                                                        std::declval<DestIt>(),
                                                                                        int{}, // actually, some callable should be here but since the function does not have any constraints any type will work
                                                                                        {3}))>> : std::true_type {};

constexpr void test_default_template_argument()
{
    static_assert(!is_replace_copy_well_formed<std::vector<not_implicitly_convertible>::iterator, std::vector<int>::iterator>::value,
                  "Input iterator value_type test: std::replace_copy must NOT have any default template argument for list-initialization");
    static_assert(!is_replace_copy_well_formed<std::vector<int>::iterator, std::vector<not_implicitly_convertible>::iterator>::value,
                  "Output iterator value_type test: std::replace_copy must NOT have any default template argument for list-initialization");
    static_assert(is_replace_copy_if_well_formed<std::vector<not_implicitly_convertible>::iterator, std::vector<int>::iterator>::value,
                  "The default template argument for list-initialization of replace_copy_if is NOT a value_type of the output iterator");
    static_assert(!is_replace_copy_if_well_formed<std::vector<int>::iterator, std::vector<not_implicitly_convertible>::iterator>::value,
                  "The default template argument for list-initialization of replace_copy_if must be a value_type of the output iterator");
}

void test_empty_list_initialization_for_replace_copy_if()
{
    {
        std::vector<int> v{3,6,0,4,0,7,8,0,3,4};
        std::vector<int> dest(v.size());
        std::vector<int> expected{0,6,0,4,0,7,8,0,0,4};
        oneapi::dpl::replace_copy_if(oneapi::dpl::execution::seq, v.begin(), v.end(), dest.begin(), [](auto x) { return x == 3; }, {});
        EXPECT_TRUE(dest == expected, "wrong effect from calling oneapi::dpl::replace_copy_if with empty list-initialized value and with `seq` policy");
    }
    {
        std::vector<int> v{3,6,0,4,0,7,8,0,3,4};
        std::vector<int> dest(v.size());
        std::vector<int> expected{0,6,0,4,0,7,8,0,0,4};
        oneapi::dpl::replace_copy_if(oneapi::dpl::execution::unseq, v.begin(), v.end(), dest.begin(), [](auto x) { return x == 3; }, {});
        EXPECT_TRUE(dest == expected, "wrong effect from calling oneapi::dpl::replace_copy_if with empty list-initialized value and with `unseq` policy");
    }

    {
        std::vector<TestUtils::DefaultInitializedToOne> v_custom{{3},{1},{5},{3},{3},{1},{8},{2},{3},{1}};
        std::vector<TestUtils::DefaultInitializedToOne> dest_custom(v_custom.size());
        std::vector<TestUtils::DefaultInitializedToOne> expected_custom{{1},{1},{5},{1},{1},{1},{8},{2},{1},{1}};
        oneapi::dpl::replace_copy_if(oneapi::dpl::execution::par, v_custom.begin(), v_custom.end(), dest_custom.begin(), [](auto x) { return x == TestUtils::DefaultInitializedToOne{3}; }, {});
        EXPECT_TRUE(dest_custom == expected_custom, "wrong effect from calling oneapi::dpl::replace_copy_if with empty list-initialized value and with `par` policy");
    }
    {
        std::vector<TestUtils::DefaultInitializedToOne> v_custom{{3},{1},{5},{3},{3},{1},{8},{2},{3},{1}};
        std::vector<TestUtils::DefaultInitializedToOne> dest_custom(v_custom.size());
        std::vector<TestUtils::DefaultInitializedToOne> expected_custom{{1},{1},{5},{1},{1},{1},{8},{2},{1},{1}};
        oneapi::dpl::replace_copy_if(oneapi::dpl::execution::par_unseq, v_custom.begin(), v_custom.end(), dest_custom.begin(), [](auto x) { return x == TestUtils::DefaultInitializedToOne{3}; }, {});
        EXPECT_TRUE(dest_custom == expected_custom, "wrong effect from calling oneapi::dpl::replace_copy_if with empty list-initialized value and with `par_unseq` policy");
    }
#if TEST_DPCPP_BACKEND_PRESENT
    std::vector<int> v{3,6,0,4,0,7,8,0,3,4};
    std::vector<int> dest(v.size());
    std::vector<int> expected{0,6,0,4,0,7,8,0,0,4};
    {
        sycl::buffer<int> buf(v);
        sycl::buffer<int> dest_buf(dest);
        oneapi::dpl::replace_copy_if(oneapi::dpl::execution::dpcpp_default, oneapi::dpl::begin(buf), oneapi::dpl::end(buf), oneapi::dpl::begin(dest_buf), [](auto x) { return x == 3; }, {});
    }
    EXPECT_TRUE(dest == expected, "wrong effect from calling oneapi::dpl::replace_copy_if with empty list-initialized value and with `device_policy` policy");
#endif
}

int
main()
{

    test<float64_t>(-666.0, 8.5, 0.33, [](const float64_t& x) { return x * x <= 1024; },
                    [](size_t j) { return ((j + 1) % 7 & 2) != 0 ? 8.5 : float64_t(j % 32 + j); });

    test<std::int32_t>(-666, 42, 99, [](const std::int32_t& x) { return x != 42; },
                  [](size_t j) { return ((j + 1) % 5 & 2) != 0 ? 42 : -1 - std::int32_t(j); });

    test<std::uint8_t>(123, 42, 99, [](const std::uint8_t& x) { return x != 42; },
                  [](size_t j) { return ((j + 1) % 5 & 2) != 0 ? 42 : 255; });


#if !TEST_DPCPP_BACKEND_PRESENT
    test<Number>(Number(42, OddTag()), Number(2001, OddTag()), Number(2017, OddTag()), IsMultiple(3, OddTag()),
                 [](std::int32_t j) { return ((j + 1) % 3 & 2) != 0 ? Number(2001, OddTag()) : Number(j, OddTag()); });
#endif

#ifdef _PSTL_TEST_REPLACE_COPY_IF
    test_algo_basic_double<std::int32_t>(run_for_rnd_fw<test_non_const<std::int32_t>>());
#endif

    test_default_template_argument();
    test_empty_list_initialization_for_replace_copy_if();

    return done();
}
