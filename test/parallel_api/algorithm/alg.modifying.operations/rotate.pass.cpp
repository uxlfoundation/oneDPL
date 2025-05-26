// -*- C++ -*-
//===-- rotate.pass.cpp ---------------------------------------------------===//
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
struct wrapper
{
    T t;
    int move_count;
    explicit wrapper(T t_) : t(t_), move_count(0) {}
    wrapper&
    operator=(const T& t_)
    {
        t = t_;
        return *this;
    }

    wrapper(const wrapper<T>& a) : move_count(0) { t = a.t; }

    wrapper<T>&
    operator=(wrapper<T>& a)
    {
        t = a.t;
        return *this;
    }

    wrapper<T>&
    operator=(wrapper<T>&& a)
    {
        t = a.t;
        move_count += 1;
        return *this;
    }
};

template <typename T>
struct compare
{
    bool
    operator()(const T& a, const T& b)
    {
        return a == b;
    }
};

template <typename T>
struct compare<wrapper<T>>
{
    bool
    operator()(const wrapper<T>& a, const wrapper<T>& b)
    {
        return a.t == b.t;
    }
};
#include <typeinfo>

template <typename Type>
struct test_one_policy
{
    template <typename ExecutionPolicy, typename Iterator, typename Size>
    void
    operator()(ExecutionPolicy&& exec, Iterator data_b, Iterator data_e, Iterator actual_b, Iterator actual_e,
               Size shift)
    {
        using namespace std;
        using T = typename iterator_traits<Iterator>::value_type;
        Iterator actual_m = ::std::next(actual_b, shift);

        copy(data_b, data_e, actual_b);
        Iterator actual_return = rotate(exec, actual_b, actual_m, actual_e);

        EXPECT_TRUE(actual_return == ::std::next(actual_b, ::std::distance(actual_m, actual_e)), "wrong result of rotate");
        auto comparator = compare<T>();
        bool check = ::std::equal(actual_return, actual_e, data_b, comparator);
        check = check && ::std::equal(actual_b, actual_return, ::std::next(data_b, shift), comparator);

        EXPECT_TRUE(check, "wrong effect of rotate");
        EXPECT_TRUE(check_move(exec, actual_b, actual_e, shift), "wrong move test of rotate");
    }

    template <typename ExecutionPolicy, typename Iterator, typename Size>
    ::std::enable_if_t<
        is_base_of_iterator_category_v<::std::random_access_iterator_tag, Iterator> &&
        !::std::is_same_v<ExecutionPolicy, oneapi::dpl::execution::sequenced_policy> &&
        ::std::is_same_v<typename ::std::iterator_traits<Iterator>::value_type, wrapper<float32_t>>,
        bool>
    check_move(ExecutionPolicy&& /* exec */, Iterator b, Iterator e, Size shift)
    {
        bool result = all_of(b, e, [](wrapper<float32_t>& a) {
            bool temp = a.move_count > 0;
            a.move_count = 0;
            return temp;
        });
        return shift == 0 || result;
    }

    template <typename ExecutionPolicy, typename Iterator, typename Size>
    ::std::enable_if_t<
        !(is_base_of_iterator_category_v<::std::random_access_iterator_tag, Iterator> &&
        !::std::is_same_v<ExecutionPolicy, oneapi::dpl::execution::sequenced_policy> &&
        ::std::is_same_v<typename ::std::iterator_traits<Iterator>::value_type, wrapper<float32_t>>),
        bool>
    check_move(ExecutionPolicy&& /* exec */, Iterator /* b */, Iterator /* e */, Size /* shift */)
    {
        return true;
    }
};

template <typename T>
void
test()
{
    const auto test_sizes = TestUtils::get_pattern_for_test_sizes();
    const std::int32_t max_len = test_sizes.back();

    Sequence<T> actual(max_len, [](::std::size_t i) { return T(i); });
    Sequence<T> data(max_len, [](::std::size_t i) { return T(i); });

    for (std::int32_t len : test_sizes)
    {
        std::int32_t shifts[] = {0, 1, 2, len / 3, (2 * len) / 3, len - 1};
        for (auto shift : shifts)
        {
            if (shift >= 0 && shift < len)
            {
                invoke_on_all_policies<>()(test_one_policy<T>(), data.begin(), data.begin() + len, actual.begin(),
                                           actual.begin() + len, shift);
            }
        }
    }
}

int
main()
{
    test<std::int8_t>();
    test<std::int16_t>();
    test<std::int32_t>();
#if !TEST_DPCPP_BACKEND_PRESENT
    test<wrapper<float64_t>>();
    test<MemoryChecker>();
    EXPECT_TRUE(MemoryChecker::alive_objects() == 0, "wrong effect from rotate: number of ctor and dtor calls is not equal");
#endif

    return done();
}
