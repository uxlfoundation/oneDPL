// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "support/test_config.h"

#include "oneapi/dpl/dynamic_selection"

#include "support/test_dynamic_selection_utils.h"
#include "support/inline_backend.h"
#include "support/utils.h"

int
main()
{
    using policy_t = oneapi::dpl::experimental::round_robin_policy<TestUtils::int_inline_backend_t>;
    std::vector<int> u{4, 5, 6, 7};
    auto f = [u](int i) { return u[(i - 1) % 4]; };

    constexpr bool just_call_submit = false;
    constexpr bool call_select_before_submit = true;

    EXPECT_EQ(0, (test_initialization<policy_t, int>(u)), "");
    EXPECT_EQ(0, (test_select<policy_t, decltype(u), decltype(f)&, false>(u, f)), "");
    EXPECT_EQ(0, (test_submit_and_wait_on_event<just_call_submit, policy_t>(u, f)), "");
    EXPECT_EQ(0, (test_submit_and_wait_on_event<call_select_before_submit, policy_t>(u, f)), "");
    EXPECT_EQ(0, (test_submit_and_wait<just_call_submit, policy_t>(u, f)), "");
    EXPECT_EQ(0, (test_submit_and_wait<call_select_before_submit, policy_t>(u, f)), "");
    EXPECT_EQ(0, (test_submit_and_wait_on_group<just_call_submit, policy_t>(u, f)), "");
    EXPECT_EQ(0, (test_submit_and_wait_on_group<call_select_before_submit, policy_t>(u, f)), "");

    return TestUtils::done();
}
