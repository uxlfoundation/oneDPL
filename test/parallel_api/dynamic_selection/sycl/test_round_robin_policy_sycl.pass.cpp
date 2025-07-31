// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "support/test_config.h"
#include "support/utils.h"

#include <iostream>
#include "oneapi/dpl/dynamic_selection"
#include "support/test_dynamic_selection_utils.h"

int
main()
{
    bool bProcessed = false;

#if TEST_DYNAMIC_SELECTION_AVAILABLE
    //using policy_t = oneapi::dpl::experimental::round_robin_policy<oneapi::dpl::experimental::sycl_backend>;
    using policy_t = oneapi::dpl::experimental::round_robin_policy<sycl::queue, oneapi::dpl::experimental::empty_extra_resource, oneapi::dpl::experimental::default_backend<sycl::queue, oneapi::dpl::experimental::empty_extra_resource>>;
    using policy_with_extra_resources_t = oneapi::dpl::experimental::round_robin_policy<sycl::queue, int, oneapi::dpl::experimental::default_backend<sycl::queue, int>>;

    std::vector<sycl::queue> u;
    build_universe(u);

    std::vector<int> v;
    for (int i = 0; i < u.size(); ++i)
    {
        v.push_back(i);
    }

    if (!u.empty())
    {
        auto n = u.size();
        std::cout << "UNIVERSE SIZE " << n << std::endl;

        auto f = [u, n](int i) { return u[(i - 1) % n]; };
        auto ef = [v, n](int i) { return v[(i - 1) % n]; };

        constexpr bool just_call_submit = false;
        constexpr bool call_select_before_submit = true;

        EXPECT_EQ(0, (test_initialization<policy_t, sycl::queue>(u)), "");
        EXPECT_EQ(0, (test_select<policy_t, decltype(u), decltype(f)&, false>(u, f)), "");
        EXPECT_EQ(0, (test_submit_and_wait_on_event<just_call_submit, policy_t>(u, f)), "");
        EXPECT_EQ(0, (test_submit_and_wait_on_event<call_select_before_submit, policy_t>(u, f)), "");
        EXPECT_EQ(0, (test_submit_and_wait<just_call_submit, policy_t>(u, f)), "");
        EXPECT_EQ(0, (test_submit_and_wait<call_select_before_submit, policy_t>(u, f)), "");
        EXPECT_EQ(0, (test_extra_resource_submit_and_wait<just_call_submit, policy_with_extra_resources_t>(u, v, f, ef)), "");
        EXPECT_EQ(0, (test_extra_resource_submit_and_wait<call_select_before_submit, policy_with_extra_resources_t>(u, v, f, ef)), "");
        EXPECT_EQ(0, (test_submit_and_wait_on_group<just_call_submit, policy_t>(u, f)), "");
        EXPECT_EQ(0, (test_submit_and_wait_on_group<call_select_before_submit, policy_t>(u, f)), "");

        bProcessed = true;
    }
#endif // TEST_DYNAMIC_SELECTION_AVAILABLE

    return TestUtils::done(bProcessed);
}
