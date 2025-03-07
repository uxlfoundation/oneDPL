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
        auto k = remove_copy(exec, first, last, out_first, value);
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
        EXPECT_TRUE(expected_count == out_count, "wrong return value from remove_copy");
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

    return done();
}
