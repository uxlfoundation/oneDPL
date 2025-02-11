// -*- C++ -*-
//===-- rotate_copy.pass.cpp ----------------------------------------------===//
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

#include <iterator>

using namespace TestUtils;

template <typename T>
struct wrapper;

template <typename T>
bool
compare(const wrapper<T>& a, const wrapper<T>& b)
{
    return a.t == b.t;
}

template <typename T>
bool
compare(const T& a, const T& b)
{
    return a == b;
}

template <typename T>
struct wrapper
{
    explicit wrapper(T t_) : t(t_) {}
    wrapper&
    operator=(const T& t_)
    {
        t = t_;
        return *this;
    }
    friend bool
    compare<T>(const wrapper<T>& a, const wrapper<T>& b);

  private:
    T t;
};

template <typename T, typename It1, typename It2>
struct comparator
{
    using T1 = typename ::std::iterator_traits<It1>::value_type;
    using T2 = typename ::std::iterator_traits<It2>::value_type;
    bool
    operator()(T1 a, T2 b)
    {
        T temp = a;
        return compare(temp, b);
    }
};

template <typename Type>
struct test_one_policy
{
    template <typename ExecutionPolicy, typename Iterator1, typename Iterator2>
    void
    operator()(ExecutionPolicy&& exec, Iterator1 data_b, Iterator1 data_e, Iterator2 actual_b, Iterator2 actual_e,
               ::std::size_t shift)
    {
        using namespace std;
        using T = typename iterator_traits<Iterator2>::value_type;
        Iterator1 data_m = ::std::next(data_b, shift);

        fill(actual_b, actual_e, T(-123));
        Iterator2 actual_return = rotate_copy(exec, data_b, data_m, data_e, actual_b);

        EXPECT_TRUE(actual_return == actual_e, "wrong result of rotate_copy");
        auto comparer = comparator<T, Iterator1, Iterator2>();
        bool check = ::std::equal(data_m, data_e, actual_b, comparer);
        check = check && ::std::equal(data_b, data_m, ::std::next(actual_b, ::std::distance(data_m, data_e)), comparer);

        EXPECT_TRUE(check, "wrong effect of rotate_copy");
    }
};

template <typename T1, typename T2>
void
test()
{
    const auto test_sizes = TestUtils::get_pattern_for_test_sizes();
    const std::int32_t max_len = test_sizes.back();

    Sequence<T2> actual(max_len, [](::std::size_t i) { return T1(i); });

    Sequence<T1> data(max_len, [](::std::size_t i) { return T1(i); });

    for (std::int32_t len : test_sizes)
    {
        std::int32_t shifts[] = {0, 1, 2, len / 3, (2 * len) / 3, len - 1};
        for (std::int32_t shift : shifts)
        {
            if (shift > 0 && shift < len)
            {
                invoke_on_all_policies<0>()(test_one_policy<T1>(), data.begin(), data.begin() + len, actual.begin(),
                                            actual.begin() + len, shift);
                invoke_on_all_policies<1>()(test_one_policy<T1>(), data.cbegin(), data.cbegin() + len, actual.begin(),
                                            actual.begin() + len, shift);
            }
        }
    }
}

int
main()
{
    test<std::int8_t, std::int8_t>();
    test<std::int16_t, std::int16_t>();
    test<std::int32_t, std::int8_t>();
    test<std::uint16_t, float32_t>();
    test<float64_t, std::int64_t>();
#if !TEST_DPCPP_BACKEND_PRESENT
    test<wrapper<float64_t>, wrapper<float64_t>>();
#endif

    return done();
}
