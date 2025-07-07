// -*- C++ -*-
//===-- header_inclusion_order_algorithm_0.pass.cpp -----------------------===//
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

#if _ENABLE_RANGES_TESTING
#include _PSTL_TEST_HEADER(execution)
#include _PSTL_TEST_HEADER(ranges)
#endif

#include "support/utils.h"

#if _ENABLE_RANGES_TESTING

struct CustomPredicate
{
    template <typename T>
    bool
    operator()(const T& value) const
    {
        return value == -1;
    }
};

template <typename Policy>
void
test_impl(Policy&& exec)
{
    using namespace oneapi::dpl::experimental::ranges;
    all_of(std::forward<Policy>(exec), views::fill(-1, 10), CustomPredicate{});
}
#endif // _ENABLE_RANGES_TESTING

int
main()
{
#if _ENABLE_RANGES_TESTING

    auto policy = TestUtils::get_dpcpp_test_policy();
    test_impl(policy);

#if TEST_CHECK_COMPILATION_WITH_DIFF_POLICY_VAL_CATEGORY
    TestUtils::check_compilation(policy, [](auto&& policy) { test_impl(std::forward<decltype(policy)>(policy)); });
#endif
#endif // _ENABLE_RANGES_TESTING

    return TestUtils::done(_ENABLE_RANGES_TESTING);
}
