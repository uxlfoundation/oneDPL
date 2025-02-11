// -*- C++ -*-
//===-- reverse.pass.cpp --------------------------------------------------===//
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
struct test_one_policy
{
#if _PSTL_ICC_18_VC141_TEST_SIMD_LAMBDA_RELEASE_BROKEN // dummy specialization by policy type, in case of broken configuration
    template <typename Iterator1, typename Iterator2>
    ::std::enable_if_t<is_base_of_iterator_category_v<::std::random_access_iterator_tag, Iterator1>>
    operator()(oneapi::dpl::execution::unsequenced_policy, Iterator1 data_b, Iterator1 data_e, Iterator2 actual_b,
               Iterator2 actual_e)
    {
    }

    template <typename Iterator1, typename Iterator2>
    ::std::enable_if_t<is_base_of_iterator_category_v<::std::random_access_iterator_tag, Iterator1>>
    operator()(oneapi::dpl::execution::parallel_unsequenced_policy, Iterator1 data_b, Iterator1 data_e, Iterator2 actual_b,
               Iterator2 actual_e)
    {
    }
#endif

    template <typename ExecutionPolicy, typename Iterator1, typename Iterator2>
    ::std::enable_if_t<is_base_of_iterator_category_v<::std::bidirectional_iterator_tag, Iterator1>>
    operator()(ExecutionPolicy&& exec, Iterator1 data_b, Iterator1 data_e, Iterator2 actual_b, Iterator2 actual_e)
    {
        using namespace std;

        copy(data_b, data_e, actual_b);

        reverse(exec, actual_b, actual_e);

        bool check = equal(data_b, data_e, reverse_iterator<Iterator2>(actual_e));

        EXPECT_TRUE(check, "wrong result of reverse");
    }

    template <typename ExecutionPolicy, typename Iterator1, typename Iterator2>
    ::std::enable_if_t<!is_base_of_iterator_category_v<::std::bidirectional_iterator_tag, Iterator1>>
    operator()(ExecutionPolicy&& /* exec */, Iterator1 /* data_b */, Iterator1 /* data_e */, Iterator2 /* actual_b */, Iterator2 /* actual_e */)
    {
    }
};

template <typename T>
void
test()
{
    const auto test_sizes = TestUtils::get_pattern_for_test_sizes();
    const std::size_t max_len = test_sizes.back();

    Sequence<T> actual(max_len);

    Sequence<T> data(max_len, [](::std::size_t i) { return T(i); });

    for (std::size_t len : test_sizes)
    {
        invoke_on_all_policies<>()(test_one_policy<T>(), data.begin(), data.begin() + len, actual.begin(),
                                   actual.begin() + len);
    }
}

template <typename T>
struct wrapper
{
    T t;
    wrapper(): t() {}
    explicit wrapper(T t_) : t(t_) {}
    bool
    operator==(const wrapper<T>& a) const
    {
        return t == a.t;
    }
};

int
main()
{
    test<std::int32_t>();
    test<std::uint8_t>();
    test<std::uint16_t>();
    test<float64_t>();
    test<wrapper<float32_t>>();

    return done();
}
