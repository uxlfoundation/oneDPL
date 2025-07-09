// -*- C++ -*-
//===-- count.pass.cpp ----------------------------------------------------===//
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

// Tests for count and count_if
#include "support/test_config.h"

#include _PSTL_TEST_HEADER(execution)
#include _PSTL_TEST_HEADER(algorithm)

#include "support/utils.h"

#if  !defined(_PSTL_TEST_COUNT) && !defined(_PSTL_TEST_COUNT_IF)
#define _PSTL_TEST_COUNT
#define _PSTL_TEST_COUNT_IF
#endif

using namespace TestUtils;

template <typename Type>
struct test_count
{
    template <typename Policy, typename Iterator, typename T>
    void
    operator()(Policy&& exec, Iterator first, Iterator last, T needle)
    {
        auto expected = ::std::count(first, last, needle);
        auto result = std::count(std::forward<Policy>(exec), first, last, needle);
        EXPECT_EQ(expected, result, "wrong count result");
    }
};

template <typename Type>
struct test_count_if
{
    template <typename Policy, typename Iterator, typename Predicate>
    void
    operator()(Policy&& exec, Iterator first, Iterator last, Predicate pred)
    {
        auto expected = ::std::count_if(first, last, pred);
        auto result = std::count_if(std::forward<Policy>(exec), first, last, pred);
        EXPECT_EQ(expected, result, "wrong count_if result");
    }
};

template <typename T>
class IsEqualPred
{
    T value;

  public:
    IsEqualPred(T value_, OddTag) : value(value_) {}
    bool
    operator()(const T& x) const
    {
        return x == value;
    }
};

template <typename In, typename T, typename Predicate, typename Convert>
void
test(T needle, Predicate pred, Convert convert)
{
    // Try sequences of various lengths.
    for (size_t n = 0; n <= 100000; n = n <= 16 ? n + 1 : size_t(3.1415 * n))
    {
        Sequence<In> in(n, [=](size_t k) -> In {
            // Sprinkle "42" and "50" early, so that short sequences have non-zero count.
            return convert((n - k - 1) % 3 == 0 ? 42 : (n - k - 2) % 5 == 0 ? 50 : 3 * (int(k) % 1000 - 500));
        });
#ifdef _PSTL_TEST_COUNT
        invoke_on_all_policies<0>()(test_count<In>(), in.begin(), in.end(), needle);
        invoke_on_all_policies<1>()(test_count<In>(), in.cbegin(), in.cend(), needle);
#endif
#ifdef _PSTL_TEST_COUNT_IF
        invoke_on_all_policies<2>()(test_count_if<In>(), in.begin(), in.end(), pred);
        invoke_on_all_policies<3>()(test_count_if<In>(), in.cbegin(), in.cend(), pred);
#endif
    }
}

struct test_non_const
{
    template <typename Policy, typename Iterator>
    void
    operator()(Policy&& exec, Iterator iter)
    {
        count_if(std::forward<Policy>(exec), iter, iter, non_const(TestUtils::IsEven<float64_t>{}));
    }
};

void test_empty_list_initialization()
{
    std::vector<int> v{3,6,0,4,0,7,8,0,3,4};
    {
        auto val = oneapi::dpl::count(oneapi::dpl::execution::seq, v.begin(), v.end(), {});
        EXPECT_TRUE(val == 3, "an empty list-initialized value is not found by oneapi::dpl::count with `seq` policy");
    }
    {
        auto val = oneapi::dpl::count(oneapi::dpl::execution::unseq, v.begin(), v.end(), {});
        EXPECT_TRUE(val == 3, "an empty list-initialized value is not found by oneapi::dpl::count with `unseq` policy");
    }

    {
        std::vector<TestUtils::DefaultInitializedToOne> v_custom{{3},{1},{5},{1},{3},{1},{8},{2},{0},{1}};
        {
            auto val = oneapi::dpl::count(oneapi::dpl::execution::par, v_custom.begin(), v_custom.end(), {});
            EXPECT_TRUE(val == 4, "an empty list-initialized value is not found by oneapi::dpl::count with `par` policy");
        }
        {
            auto val = oneapi::dpl::count(oneapi::dpl::execution::par_unseq, v_custom.begin(), v_custom.end(), {});
            EXPECT_TRUE(val == 4, "an empty list-initialized value is not found by oneapi::dpl::count with `par_unseq` policy");
        }
    }
#if TEST_DPCPP_BACKEND_PRESENT
    sycl::buffer<int> buf(v);
    auto val = oneapi::dpl::count(oneapi::dpl::execution::dpcpp_default, oneapi::dpl::begin(buf), oneapi::dpl::end(buf), {});
    EXPECT_TRUE(val == 3, "an empty list-initialized value is not found by oneapi::dpl::count with `device_policy` policy");
#endif
}

int
main()
{
    test<std::int16_t>(42, IsEqualPred<std::int16_t>(50, OddTag()), [](std::int16_t j) { return j; });
    test<std::int32_t>(42, [](const std::int32_t&) { return true; }, [](std::int32_t j) { return j; });
    test<float64_t>(42, IsEqualPred<float64_t>(50, OddTag()), [](std::int32_t j) { return float64_t(j); });
#if !TEST_DPCPP_BACKEND_PRESENT
    test<Number>(Number(42, OddTag()), IsEqualPred<Number>(Number(50, OddTag()), OddTag()),
                 [](std::int32_t j) { return Number(j, OddTag()); });
#endif
#ifdef _PSTL_TEST_COUNT_IF
    test_algo_basic_single<std::int32_t>(run_for_rnd_fw<test_non_const>());
#endif

    test_empty_list_initialization();

    return done();
}
