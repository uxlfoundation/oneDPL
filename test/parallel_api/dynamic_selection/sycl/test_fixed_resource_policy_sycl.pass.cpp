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
#include "support/utils.h"

int
main()
{
    bool bProcessed = false;

#if TEST_DYNAMIC_SELECTION_AVAILABLE
    using policy_t = oneapi::dpl::experimental::fixed_resource_policy<sycl::queue, oneapi::dpl::experimental::default_backend<sycl::queue>>;
    std::vector<sycl::queue> u;
    build_universe(u);
    if (!u.empty())
    {
        auto f = [u](int i, int offset = 0) { return u[offset]; };

        constexpr bool just_call_submit = false;
        constexpr bool call_select_before_submit = true;

        auto actual = test_initialization<policy_t, sycl::queue>(u);
        actual = test_select<policy_t, decltype(u), decltype(f)&, false>(u, f);
        actual = test_submit_and_wait_on_event<just_call_submit, policy_t>(u, f);
        actual = test_submit_and_wait_on_event<just_call_submit, policy_t>(u, f, 1);
        actual = test_submit_and_wait_on_event<call_select_before_submit, policy_t>(u, f);
        actual = test_submit_and_wait_on_event<call_select_before_submit, policy_t>(u, f, 1);
        actual = test_submit_and_wait<just_call_submit, policy_t>(u, f);
        actual = test_submit_and_wait<just_call_submit, policy_t>(u, f, 1);
        actual = test_submit_and_wait<call_select_before_submit, policy_t>(u, f);
        actual = test_submit_and_wait<call_select_before_submit, policy_t>(u, f, 1);
        actual = test_submit_and_wait_on_group<just_call_submit, policy_t>(u, f);
        actual = test_submit_and_wait_on_group<just_call_submit, policy_t>(u, f, 1);
        actual = test_submit_and_wait_on_group<call_select_before_submit, policy_t>(u, f);
        actual = test_submit_and_wait_on_group<call_select_before_submit, policy_t>(u, f, 1);

        bProcessed = true;
    }
#endif // TEST_DYNAMIC_SELECTION_AVAILABLE

    return TestUtils::done(bProcessed);
}
