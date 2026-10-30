// -*- C++ -*-
//===-- search_n.pass.cpp -------------------------------------------------===//
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

#include <limits>
#include <vector>

using namespace TestUtils;

template <typename Type>
struct test_search_n
{
    template <typename ExecutionPolicy, typename Iterator, typename Size, typename T, typename Predicate>
    void
    operator()(ExecutionPolicy&& exec, Iterator b, Iterator e, Size count, const T& value, Predicate pred)
    {
        using namespace std;
        auto expected = search_n(b, e, count, value, pred);
        auto actual = search_n(std::forward<ExecutionPolicy>(exec), b, e, count, value);
        EXPECT_EQ(expected, actual, "wrong return result from search_n");
    }
};

// The default predicate must compare the elements with the value as is ([alg.search]),
// without converting the value to the value type of the sequence
template <typename Type>
struct test_search_n_mixed_types
{
    template <typename ExecutionPolicy, typename Iterator, typename Size, typename T>
    void
    operator()(ExecutionPolicy&& exec, Iterator b, Iterator e, Size count, const T& value)
    {
        auto expected = std::search_n(b, e, count, value);
        auto actual = search_n(std::forward<ExecutionPolicy>(exec), b, e, count, value);
        EXPECT_EQ(expected, actual, "wrong return result from search_n with mixed types");
    }
};

template <typename Type>
struct test_search_n_predicate
{
    template <typename ExecutionPolicy, typename Iterator, typename Size, typename T, typename Predicate>
    void
    operator()(ExecutionPolicy&& exec, Iterator b, Iterator e, Size count, const T& value, Predicate pred)
    {
        using namespace std;
        auto expected = search_n(b, e, count, value, pred);
        auto actual = search_n(std::forward<ExecutionPolicy>(exec), b, e, count, value, pred);
        EXPECT_EQ(expected, actual, "wrong return result from search_n with a predicate");
    }
};

template <typename T>
void
test()
{

    const ::std::size_t max_n1 = 100000;
    const T value = T(1);
    for (::std::size_t n1 = 0; n1 <= max_n1; n1 = n1 <= 16 ? n1 + 1 : size_t(3.1415 * n1))
    {
        ::std::size_t sub_n[] = {0, 1, 3, n1, (n1 * 10) / 8};
        ::std::size_t res[] = {0, 1, n1 / 2, n1};
        for (auto n2 : sub_n)
        {
            for (auto r : res)
            {
                Sequence<T> in(n1, [](::std::size_t) { return T(0); });
                ::std::size_t i = r, isub = 0;
                for (; i < n1 && isub < n2; ++i, ++isub)
                    in[i] = value;

                invoke_on_all_policies<0>()(test_search_n<T>(), in.begin(), in.begin() + n1, n2, value,
                                            ::std::equal_to<T>());
                invoke_on_all_policies<1>()(test_search_n_predicate<T>(), in.begin(), in.begin() + n1, n2, value,
                                            ::std::equal_to<T>());
#if !ONEDPL_FPGA_DEVICE
                invoke_on_all_policies<2>()(test_search_n<T>(), in.cbegin(), in.cbegin() + n1, n2, value,
                                            ::std::equal_to<T>());
                invoke_on_all_policies<3>()(test_search_n_predicate<T>(), in.cbegin(), in.cbegin() + n1, n2, value,
                                            ::std::equal_to<T>());
#endif
            }
        }
    }
}

void
test_mixed_types()
{
    const std::size_t n = 1000;
    Sequence<std::int32_t> in(n, [](std::size_t) { return std::int32_t(2); });

    // 2.5 is not equal to any element; converting it to std::int32_t would give a match at the beginning
    invoke_on_all_policies<4>()(test_search_n_mixed_types<std::int32_t>(), in.begin(), in.end(), 2, 2.5);
    // 2.0 is equal to the elements
    invoke_on_all_policies<5>()(test_search_n_mixed_types<std::int32_t>(), in.begin(), in.end(), 2, 2.0);
}

// The range length and the count must be compared as is, not after narrowing the length to the type of the count
template <typename Size, int CallNumber>
void
test_narrow_count_type()
{
    const std::size_t type_range = std::size_t(std::numeric_limits<Size>::max()) + 1;
    for (std::size_t n : {type_range + 1, 2 * type_range, 3 * type_range + 1})
    {
        for (Size count : {Size(1), Size(2)})
        {
            for (std::size_t pos : {std::size_t(0), n / 2, n - count})
            {
                Sequence<std::int32_t> in(n, [](std::size_t) { return 0; });
                for (std::size_t i = 0; i < count; ++i)
                    in[pos + i] = 1;

                invoke_on_all_policies<CallNumber>()(test_search_n<std::int32_t>(), in.begin(), in.end(), count,
                                                     std::int32_t(1), std::equal_to<std::int32_t>());
            }
        }
    }
}

// A count type that converts implicitly from and to long has no narrowing problem and must still compile
struct implicit_count
{
    implicit_count(long value) : _value(value) {}
    operator long() const { return _value; }

  private:
    long _value;
};

void
test_wrapper_count_type()
{
    std::vector<std::int32_t> in(10, 0);
    in[3] = in[4] = 1;

    namespace dpl_exec = oneapi::dpl::execution;
    EXPECT_TRUE(search_n(dpl_exec::par, in.begin(), in.end(), implicit_count(2), 1) == in.begin() + 3,
                "wrong result of search_n(par) with a count wrapper type");
    EXPECT_TRUE(search_n(dpl_exec::unseq, in.begin(), in.end(), implicit_count(2), 1) == in.begin() + 3,
                "wrong result of search_n(unseq) with a count wrapper type");
    EXPECT_TRUE(search_n(dpl_exec::par_unseq, in.begin(), in.end(), implicit_count(2), 1) == in.begin() + 3,
                "wrong result of search_n(par_unseq) with a count wrapper type");
}

template <typename T>
struct test_non_const
{
    template <typename Policy, typename Iterator>
    void
    operator()(Policy&& exec, Iterator iter)
    {
        search_n(std::forward<Policy>(exec), iter, iter, 0, T(0), non_const(std::equal_to<T>()));
    }
};

void test_empty_list_initialization()
{
    std::vector<int> v{3,6,5,4,0,0,0,0,2,4};
    {
        auto it = oneapi::dpl::search_n(oneapi::dpl::execution::seq, v.begin(), v.end(), 4, {});
        EXPECT_TRUE(it == (v.begin() + 4), "an empty list-initialized value is not found by oneapi::dpl::search_n with `seq` policy");
    }
    {
        auto it = oneapi::dpl::search_n(oneapi::dpl::execution::seq, v.begin(), v.end(), 4, {}, std::equal_to{});
        EXPECT_TRUE(it == (v.begin() + 4), "an empty list-initialized value is not found by oneapi::dpl::search_n with `seq` policy with predicate");
    }
    {
        auto it = oneapi::dpl::search_n(oneapi::dpl::execution::unseq, v.begin(), v.end(), 4, {});
        EXPECT_TRUE(it == (v.begin() + 4), "an empty list-initialized value is not found by oneapi::dpl::search_n with `unseq` policy");
    }
    {
        auto it = oneapi::dpl::search_n(oneapi::dpl::execution::unseq, v.begin(), v.end(), 4, {}, std::equal_to{});
        EXPECT_TRUE(it == (v.begin() + 4), "an empty list-initialized value is not found by oneapi::dpl::search_n with `unseq` policy with predicate");
    }

    {
        std::vector<TestUtils::DefaultInitializedToOne> v_custom{{3},{6},{5},{4},{3},{1},{1},{1},{1},{4}};
        {
            auto it = oneapi::dpl::search_n(oneapi::dpl::execution::par, v_custom.begin(), v_custom.end(), 4, {});
            EXPECT_TRUE(it == (v_custom.begin() + 5), "an empty list-initialized value is not found by oneapi::dpl::search_n with `par` policy");
        }
        {
            auto it = oneapi::dpl::search_n(oneapi::dpl::execution::par, v_custom.begin(), v_custom.end(), 4, {}, std::equal_to{});
            EXPECT_TRUE(it == (v_custom.begin() + 5), "an empty list-initialized value is not found by oneapi::dpl::search_n with `par` policy with predicate");
        }
        {
            auto it = oneapi::dpl::search_n(oneapi::dpl::execution::par_unseq, v_custom.begin(), v_custom.end(), 4, {});
            EXPECT_TRUE(it == (v_custom.begin() + 5), "an empty list-initialized value is not found by oneapi::dpl::search_n with `par_unseq` policy");
        }
        {
            auto it = oneapi::dpl::search_n(oneapi::dpl::execution::par_unseq, v_custom.begin(), v_custom.end(), 4, {}, std::equal_to{});
            EXPECT_TRUE(it == (v_custom.begin() + 5), "an empty list-initialized value is not found by oneapi::dpl::search_n with `par_unseq` policy with predicate");
        }
    }
#if TEST_DPCPP_BACKEND_PRESENT
    sycl::buffer<int> buf(v);
    auto it = oneapi::dpl::search_n(oneapi::dpl::execution::dpcpp_default, oneapi::dpl::begin(buf), oneapi::dpl::end(buf), 4, {});
    EXPECT_TRUE(it.get_idx() == 4, "an empty list-initialized value is not found by oneapi::dpl::search_n with `device_policy` policy");
#endif
}

int
main()
{
    test<std::int32_t>();
#if !ONEDPL_FPGA_DEVICE
    test<std::uint16_t>();
#endif
    test<float64_t>();
    test<bool>();

    test_algo_basic_single<std::int32_t>(run_for_rnd_fw<test_non_const<std::int32_t>>());

    test_empty_list_initialization();
    test_mixed_types();
    test_narrow_count_type<std::uint8_t, 6>();
    test_narrow_count_type<std::uint16_t, 7>();
    test_wrapper_count_type();

    return done();
}
