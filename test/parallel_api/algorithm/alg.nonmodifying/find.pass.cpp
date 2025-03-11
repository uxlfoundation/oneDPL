// -*- C++ -*-
//===-- find.pass.cpp -----------------------------------------------------===//
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

// Tests for find
#include "support/test_config.h"

#include _PSTL_TEST_HEADER(execution)
#include _PSTL_TEST_HEADER(algorithm)

#include "support/utils.h"

using namespace TestUtils;

template <typename T>
struct test_find
{
    template <typename Policy, typename Iterator, typename Value>
    void
    operator()(Policy&& exec, Iterator first, Iterator last, Value value)
    {
        auto i = ::std::find(first, last, value);
        auto j = find(exec, first, last, value);
        EXPECT_TRUE(i == j, "wrong return value from find");
    }
};

template <typename T, typename Value, typename Hit, typename Miss>
void
test(Value value, Hit hit, Miss miss)
{
    // Try sequences of various lengths.
    for (size_t n = 0; n <= 100000; n = n <= 16 ? n + 1 : size_t(3.1415 * n))
    {
        Sequence<T> in(n, [&](size_t k) -> T { return miss(n ^ k); });
        // Try different find positions, including not found.
        // By going backwards, we can add extra matches that are *not* supposed to be found.
        // The decreasing exponential gives us O(n) total work for the loop since each find takes O(m) time.
        for (size_t m = n; m > 0; m *= 0.6)
        {
            if (m < n)
                in[m] = hit(n ^ m);
            invoke_on_all_policies<0>()(test_find<T>(), in.begin(), in.end(), value);
            invoke_on_all_policies<1>()(test_find<T>(), in.cbegin(), in.cend(), value);
        }
    }
}

// Type defined for sake of checking that ::std::find works with asymmetric ==.
class Weird
{
    Number value;

  public:
    friend bool
    operator==(Number x, Weird y)
    {
        return x == y.value;
    }
    Weird(std::int32_t val, OddTag) : value(val, OddTag()) {}
};

int
main()
{
    // Note that the "hit" and "miss" functions here avoid overflow issues.
#if !TEST_DPCPP_BACKEND_PRESENT
    test<Number>(Weird(42, OddTag()), [](std::int32_t) { return Number(42, OddTag()); }, // hit
                 [](std::int32_t j) { return Number(j == 42 ? 0 : j, OddTag()); });      // miss
#endif

    // Test with value that is equal to two different bit patterns (-0.0 and 0.0)
    test<float32_t>(-0.0, [](std::int32_t j) { return j & 1 ? 0.0 : -0.0; }, // hit
                    [](std::int32_t j) { return j == 0 ? ~j : j; });         // miss


    return done();
}
