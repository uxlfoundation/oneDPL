// -*- C++ -*-
//===-- remove_copy.pass.cpp ----------------------------------------------===//
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
struct run_remove_copy
{
    template <typename Policy, typename InputIterator, typename OutputIterator, typename OutputIterator2, typename Size>
    void
    operator()(Policy&& exec, InputIterator first, InputIterator last, OutputIterator out_first,
               OutputIterator
#if !TEST_DPCPP_BACKEND_PRESENT
               out_last
#endif
               , OutputIterator2 expected_first, OutputIterator2 /* expected_last */, Size n,
               const T& value, T trash)
    {
        // Cleaning
        ::std::fill_n(expected_first, n, trash);
        ::std::fill_n(out_first, n, trash);

        // Run copy_if
        [[maybe_unused]] auto i = remove_copy(first, last, expected_first, value);
        auto k = remove_copy(std::forward<Policy>(exec), first, last, out_first, value);
#if !TEST_DPCPP_BACKEND_PRESENT
        EXPECT_EQ_N(expected_first, out_first, n, "wrong remove_copy effect");
        for (size_t j = 0; j < GuardSize; ++j)
        {
            ++k;
        }
        EXPECT_TRUE(out_last == k, "wrong return value from remove_copy");
#else
        auto expected_count = ::std::distance(expected_first, i);
        auto out_count = ::std::distance(out_first, k);
        EXPECT_EQ(expected_count, out_count, "wrong return value from remove_copy");
        EXPECT_EQ_N(expected_first, out_first, expected_count, "wrong remove_copy effect");
#endif
    }
};

template <typename T, typename Convert>
void
test(T trash, const T& value, Convert convert, bool check_weakness = true)
{
    // Try sequences of various lengths.
    for (size_t n = 0; n <= 100000; n = n <= 16 ? n + 1 : size_t(3.1415 * n))
    {
#if !TEST_DPCPP_BACKEND_PRESENT
        // count is number of output elements, plus a handful
        // more for sake of detecting buffer overruns.
        size_t count = GuardSize;
#else
        size_t count = n;
#endif
        Sequence<T> in(n, [&](size_t k) -> T {
            T x = convert(n ^ k);
#if !TEST_DPCPP_BACKEND_PRESENT
            count += !(x == value) ? 1 : 0;
#endif
            return x;
        });
        using namespace std;

        Sequence<T> out(count, [=](size_t) { return trash; });
        Sequence<T> expected(count, [=](size_t) { return trash; });
        if (check_weakness)
        {
            auto expected_result = remove_copy(in.cfbegin(), in.cfend(), expected.begin(), value);
            size_t m = expected_result - expected.begin();
            EXPECT_TRUE(n / 4 <= m && m <= 3 * (n + 1) / 4, "weak test for remove_copy");
        }
        invoke_on_all_policies<0>()(run_remove_copy<T>(), in.begin(), in.end(), out.begin(), out.end(),
                                    expected.begin(), expected.end(), count, value, trash);
        invoke_on_all_policies<1>()(run_remove_copy<T>(), in.cbegin(), in.cend(), out.begin(), out.end(),
                                    expected.begin(), expected.end(), count, value, trash);
    }
}

struct not_implicitly_convertible
{
    explicit not_implicitly_convertible(int v) {}
};

template <typename It, typename DestIt, typename Void = void>
struct is_remove_copy_well_formed : std::false_type {};

template <typename It, typename DestIt>
struct is_remove_copy_well_formed<It, DestIt,
                                  std::void_t<decltype(oneapi::dpl::remove_copy(oneapi::dpl::execution::seq,
                                                                                std::declval<It>(),
                                                                                std::declval<It>(),
                                                                                std::declval<DestIt>(),
                                                                                {3}))>> : std::true_type {};

constexpr void test_default_template_argument_from_output_iterator()
{
    static_assert(is_remove_copy_well_formed<std::vector<int>::iterator, std::vector<not_implicitly_convertible>::iterator>::value,
                  "The default template argument for list-initialization of remove_copy is NOT a value_type of the input iterator");
    static_assert(!is_remove_copy_well_formed<std::vector<not_implicitly_convertible>::iterator, std::vector<int>::iterator>::value,
                  "The default template argument for list-initialization of remove_copy must be a value_type of the input iterator");
}

void test_empty_list_initialization()
{
    {
        std::vector<int> v{3,6,0,4,0,7,8,0,3,4};
        std::vector<int> dest(v.size());
        std::vector<int> expected{3,6,4,7,8,3,4};
        auto it = oneapi::dpl::remove_copy(oneapi::dpl::execution::seq, v.begin(), v.end(), dest.begin(), {});
        EXPECT_TRUE(it == dest.begin() + 7, "not all empty list-initialized values are properly remove_copy by oneapi::dpl::remove_copy with `seq` policy");
        dest.erase(it, dest.end());
        EXPECT_TRUE(dest == expected, "wrong effect from calling oneapi::dpl::remove_copy with empty list-initialized value and with `seq` policy");
    }
    {
        std::vector<int> v{3,6,0,4,0,7,8,0,3,4};
        std::vector<int> dest(v.size());
        std::vector<int> expected{3,6,4,7,8,3,4};
        auto it = oneapi::dpl::remove_copy(oneapi::dpl::execution::unseq, v.begin(), v.end(), dest.begin(), {});
        EXPECT_TRUE(it == dest.begin() + 7, "not all empty list-initialized values are properly remove_copy by oneapi::dpl::remove_copy with `unseq` policy");
        dest.erase(it, dest.end());
        EXPECT_TRUE(dest == expected, "wrong effect from calling oneapi::dpl::remove_copy with empty list-initialized value and with `unseq` policy");
    }

    {
        {
            std::vector<TestUtils::DefaultInitializedToOne> v_custom{{3},{1},{5},{1},{3},{1},{8},{2},{0},{1}};
            std::vector<TestUtils::DefaultInitializedToOne> dest_custom{v_custom.size()};
            std::vector<TestUtils::DefaultInitializedToOne> expected_custom{{3},{5},{3},{8},{2},{0}};
            auto it = oneapi::dpl::remove_copy(oneapi::dpl::execution::par, v_custom.begin(), v_custom.end(), dest_custom.begin(), {});
            EXPECT_TRUE(it == dest_custom.begin() + 6, "not all empty list-initialized values are properly remove_copy by oneapi::dpl::remove_copy with `par` policy");
            dest_custom.erase(it, dest_custom.end());
            EXPECT_TRUE(dest_custom == expected_custom, "wrong effect from calling oneapi::dpl::remove_copy with empty list-initialized value and with `par` policy");
        }
        {
            std::vector<TestUtils::DefaultInitializedToOne> v_custom{{3},{1},{5},{1},{3},{1},{8},{2},{0},{1}};
            std::vector<TestUtils::DefaultInitializedToOne> dest_custom{v_custom.size()};
            std::vector<TestUtils::DefaultInitializedToOne> expected_custom{{3},{5},{3},{8},{2},{0}};
            auto it = oneapi::dpl::remove_copy(oneapi::dpl::execution::par_unseq, v_custom.begin(), v_custom.end(), dest_custom.begin(), {});
            EXPECT_TRUE(it == dest_custom.begin() + 6, "not all empty list-initialized values are properly remove_copy by oneapi::dpl::remove_copy with `par_unseq` policy");
            dest_custom.erase(it, dest_custom.end());
            EXPECT_TRUE(dest_custom == expected_custom, "wrong effect from calling oneapi::dpl::remove_copy with empty list-initialized value and with `par_unseq` policy");
        }
    }
#if TEST_DPCPP_BACKEND_PRESENT
    std::vector<int> v{3,6,0,4,0,7,8,0,3,4};
    std::vector<int> dest(v.size());
    std::vector<int> expected{3,6,4,7,8,3,4};
    sycl::buffer<int> buf(v);
    sycl::buffer<int> dest_buf(v);
    auto it = oneapi::dpl::remove_copy(oneapi::dpl::execution::dpcpp_default, oneapi::dpl::begin(buf), oneapi::dpl::end(buf), oneapi::dpl::begin(dest_buf), {});
    EXPECT_TRUE(it.get_idx() == 7, "not all empty list-initialized values are properly remove_copy by oneapi::dpl::remove_copy with `device_policy` policy");
    dest.erase(dest.begin() + it.get_idx(), dest.end());
    EXPECT_TRUE(dest == expected, "wrong effect from calling oneapi::dpl::remove_copy with empty list-initialized value and with `device_policy` policy");
#endif
}

int
main()
{
#if !ONEDPL_FPGA_DEVICE
    test<float64_t>(-666.0, 8.5, [](size_t j) { return ((j + 1) % 7 & 2) != 0 ? 8.5 : float64_t(j % 32 + j); });
#endif

    test<std::int32_t>(-666, 42, [](size_t j) { return ((j + 1) % 5 & 2) != 0 ? 42 : -1 - std::int32_t(j); });

#if !TEST_DPCPP_BACKEND_PRESENT
    test<Number>(Number(42, OddTag()), Number(2001, OddTag()),
                 [](std::int32_t j) { return ((j + 1) % 3 & 2) != 0 ? Number(2001, OddTag()) : Number(j, OddTag()); });
#endif

    test_default_template_argument_from_output_iterator();
    test_empty_list_initialization();

    return done();
}
