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

#include "support/inline_backend.h"
#include "support/test_offset_utils.h"
#include "support/utils.h"

int
main()
{
    //tests using default backend and only a resource type. (no user backend provided)
    using policy_t = oneapi::dpl::experimental::fixed_resource_policy<int, oneapi::dpl::identity, oneapi::dpl::experimental::default_backend<int>>;
    std::vector<int> u{4, 5, 6, 7};
    auto f = [u](size_t, size_t offset = 0) { return u[offset]; };

    constexpr bool just_call_submit = false;
    constexpr bool call_select_before_submit = true;

    test_initialization<policy_t, int>(u);
    test_select<policy_t, decltype(u), decltype(f)&, false>(u, f);
    test_submit_and_wait_on_event<just_call_submit, policy_t>(u, f);
    test_submit_and_wait_on_event<just_call_submit, policy_t>(u, f, 1);
    test_submit_and_wait_on_event<just_call_submit, policy_t>(u, f, 2);
    test_submit_and_wait_on_event<just_call_submit, policy_t>(u, f, 3);
    test_submit_and_wait_on_event<call_select_before_submit, policy_t>(u, f);
    test_submit_and_wait_on_event<call_select_before_submit, policy_t>(u, f, 1);
    test_submit_and_wait_on_event<call_select_before_submit, policy_t>(u, f, 2);
    test_submit_and_wait_on_event<call_select_before_submit, policy_t>(u, f, 3);
    test_submit_and_wait<just_call_submit, policy_t>(u, f);
    test_submit_and_wait<just_call_submit, policy_t>(u, f, 1);
    test_submit_and_wait<just_call_submit, policy_t>(u, f, 2);
    test_submit_and_wait<just_call_submit, policy_t>(u, f, 3);
    test_submit_and_wait<call_select_before_submit, policy_t>(u, f);
    test_submit_and_wait<call_select_before_submit, policy_t>(u, f, 1);
    test_submit_and_wait<call_select_before_submit, policy_t>(u, f, 2);
    test_submit_and_wait<call_select_before_submit, policy_t>(u, f, 3);
    /*
    test_submit_and_wait_on_group<just_call_submit, policy_t>(u, f);
    test_submit_and_wait_on_group<just_call_submit, policy_t>(u, f, 1);
    test_submit_and_wait_on_group<just_call_submit, policy_t>(u, f, 2);
    test_submit_and_wait_on_group<just_call_submit, policy_t>(u, f, 3);
    test_submit_and_wait_on_group<call_select_before_submit, policy_t>(u, f);
    test_submit_and_wait_on_group<call_select_before_submit, policy_t>(u, f, 1);
    test_submit_and_wait_on_group<call_select_before_submit, policy_t>(u, f, 2);
    test_submit_and_wait_on_group<call_select_before_submit, policy_t>(u, f, 3);
    */

    using policy1_t = oneapi::dpl::experimental::fixed_resource_policy<DummyResource, oneapi::dpl::identity, oneapi::dpl::experimental::default_backend<DummyResource>>;
    std::vector<DummyResource> u1;
    auto f1 = [u1](size_t, size_t offset = 0) { return u1[offset]; };
     for (int i=4; i<7; ++i) {
        u1.push_back(DummyResource(i));
    }

    test_initialization<policy1_t, DummyResource>(u1);
    test_select<policy1_t, decltype(u1), decltype(f1)&, false>(u1, f1);
    test_submit_and_wait_on_event<just_call_submit, policy1_t>(u1, f1);
    test_submit_and_wait_on_event<just_call_submit, policy1_t>(u1, f1, 1);
    /*
    test_submit_and_wait_on_event<just_call_submit, policy1_t>(u1, f1, 2);
    test_submit_and_wait_on_event<just_call_submit, policy1_t>(u1, f1, 3);
    test_submit_and_wait_on_event<call_select_before_submit, policy1_t>(u1, f1);
    test_submit_and_wait_on_event<call_select_before_submit, policy1_t>(u1, f1, 1);
    test_submit_and_wait_on_event<call_select_before_submit, policy1_t>(u1, f1, 2);
    test_submit_and_wait_on_event<call_select_before_submit, policy1_t>(u1, f1, 3);
    test_submit_and_wait<just_call_submit, policy1_t>(u1, f1);
    test_submit_and_wait<just_call_submit, policy1_t>(u1, f1, 1);
    test_submit_and_wait<just_call_submit, policy1_t>(u1, f1, 2);
    test_submit_and_wait<just_call_submit, policy1_t>(u1, f1, 3);
    test_submit_and_wait<call_select_before_submit, policy1_t>(u1, f1);
    test_submit_and_wait<call_select_before_submit, policy1_t>(u1, f1, 1);
    test_submit_and_wait<call_select_before_submit, policy1_t>(u1, f1, 2);
    test_submit_and_wait<call_select_before_submit, policy1_t>(u1, f1, 3);
    test_submit_and_wait_on_group<just_call_submit, policy1_t>(u1, f1);
    test_submit_and_wait_on_group<just_call_submit, policy1_t>(u1, f1, 1);
    test_submit_and_wait_on_group<just_call_submit, policy1_t>(u1, f1, 2);
    test_submit_and_wait_on_group<just_call_submit, policy1_t>(u1, f1, 3);
    test_submit_and_wait_on_group<call_select_before_submit, policy1_t>(u1, f1);
    test_submit_and_wait_on_group<call_select_before_submit, policy1_t>(u1, f1, 1);
    test_submit_and_wait_on_group<call_select_before_submit, policy1_t>(u1, f1, 2);
    test_submit_and_wait_on_group<call_select_before_submit, policy1_t>(u1, f1, 3);
*/
    return TestUtils::done();
}
