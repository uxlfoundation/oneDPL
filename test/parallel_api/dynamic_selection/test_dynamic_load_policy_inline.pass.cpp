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
    //tests using default backend and only a resource type. (no user backend provided)
    using policy_t = oneapi::dpl::experimental::dynamic_load_policy<int, oneapi::dpl::identity, oneapi::dpl::experimental::default_backend<int>>;
    std::vector<int> u{4, 5, 6, 7};

    // should always pick the "offset" device since executed inline
    // there is no overlap and so "offset" is always unloaded at selection time
    auto f = [u](int) { return u[0]; };

    constexpr bool just_call_submit = false;
    constexpr bool call_select_before_submit = true;

    EXPECT_EQ(0, (test_initialization<policy_t, int>(u)), "");
    EXPECT_EQ(0, (test_select<policy_t, decltype(u), decltype(f)&, false>(u, f)), "");
    EXPECT_EQ(0, (test_submit_and_wait_on_event<just_call_submit, policy_t>(u, f)), "");
    EXPECT_EQ(0, (test_submit_and_wait_on_event<call_select_before_submit, policy_t>(u, f)), "");
    EXPECT_EQ(0, (test_submit_and_wait<just_call_submit, policy_t>(u, f)), "");
    EXPECT_EQ(0, (test_submit_and_wait<call_select_before_submit, policy_t>(u, f)), "");

    //tests using minimal backend that only provides a wait functionality through the resource
    using policy1_t = oneapi::dpl::experimental::dynamic_load_policy<DummyResource, oneapi::dpl::identity, oneapi::dpl::experimental::default_backend<DummyResource>>;
    std::vector<DummyResource> u1;
    for (int i=0; i<4; ++i) {
        u1.push_back(DummyResource(i));
    }
    auto f1 = [u1](int) { return u1[0]; };

    EXPECT_EQ(0, (test_initialization<policy1_t, DummyResource>(u1)), "");
    EXPECT_EQ(0, (test_select<policy1_t, decltype(u1), decltype(f1)&, false>(u1, f1)), "");
    EXPECT_EQ(0, (test_submit_and_wait_on_event<just_call_submit, policy1_t>(u1, f1)), "");
    EXPECT_EQ(0, (test_submit_and_wait_on_event<call_select_before_submit, policy1_t>(u1, f1)), "");
    EXPECT_EQ(0, (test_submit_and_wait<just_call_submit, policy1_t>(u1, f1)), "");
    EXPECT_EQ(0, (test_submit_and_wait<call_select_before_submit, policy1_t>(u1, f1)), "");
    EXPECT_EQ(0, (test_submit_and_wait_on_group<just_call_submit, policy1_t>(u1, f1)), "");
    EXPECT_EQ(0, (test_submit_and_wait_on_group<call_select_before_submit, policy1_t>(u1, f1)), "");

    return TestUtils::done();
}
