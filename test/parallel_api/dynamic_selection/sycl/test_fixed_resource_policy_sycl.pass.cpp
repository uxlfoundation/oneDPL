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
    using policy_t = oneapi::dpl::experimental::fixed_resource_policy<sycl::queue, std::identity, oneapi::dpl::experimental::default_backend<sycl::queue, std::identity>>;
    std::vector<sycl::queue> u;
    build_universe(u);
    if (!u.empty())
    {
        auto f = [u](int, int offset = 0) { return u[offset]; };

        
        constexpr bool just_call_submit = false;
        constexpr bool call_select_before_submit = true;
        
        EXPECT_EQ(0, (test_initialization<policy_t, sycl::queue>(u)), "");
        EXPECT_EQ(0, (test_select<policy_t, decltype(u), decltype(f)&, false>(u, f)), "");
        EXPECT_EQ(0, (test_submit_and_wait_on_event<just_call_submit, policy_t>(u, f)), "");
        EXPECT_EQ(0, (test_submit_and_wait_on_event<call_select_before_submit, policy_t>(u, f)), "");
        EXPECT_EQ(0, (test_submit_and_wait<just_call_submit, policy_t>(u, f)), "");
        EXPECT_EQ(0, (test_submit_and_wait<call_select_before_submit, policy_t>(u, f)), "");
        EXPECT_EQ(0, (test_submit_and_wait_on_group<just_call_submit, policy_t>(u, f)), "");
        EXPECT_EQ(0, (test_submit_and_wait_on_group<call_select_before_submit, policy_t>(u, f)), "");
        
        auto deref_op = [](auto pointer){return *pointer;};
        using policy_pointer_t = oneapi::dpl::experimental::fixed_resource_policy<sycl::queue*, decltype(deref_op), oneapi::dpl::experimental::default_backend<sycl::queue*, decltype(deref_op)>>;
        
        std::vector<sycl::queue*> u_ptrs;
        u_ptrs.reserve(u.size());
        for (auto& e: u)
        {
            u_ptrs.push_back(&e);
        }
        auto f_ptrs = [u_ptrs](int, int offset = 0) { return u_ptrs[offset]; };

        EXPECT_EQ(0, (test_initialization<policy_pointer_t, sycl::queue*>(u_ptrs, deref_op)), "");
        EXPECT_EQ(0, (test_select<policy_pointer_t, decltype(u_ptrs), decltype(f_ptrs)&, false>(u_ptrs, f_ptrs, deref_op)), "");
        EXPECT_EQ(0, (test_submit_and_wait_on_event<just_call_submit, policy_pointer_t>(u_ptrs, f_ptrs, deref_op)), "");
        EXPECT_EQ(0, (test_submit_and_wait_on_event<call_select_before_submit, policy_pointer_t>(u_ptrs, f_ptrs, deref_op)), "");
        EXPECT_EQ(0, (test_submit_and_wait<just_call_submit, policy_pointer_t>(u_ptrs, f_ptrs, deref_op)), "");
        EXPECT_EQ(0, (test_submit_and_wait<call_select_before_submit, policy_pointer_t>(u_ptrs, f_ptrs, deref_op)), "");
        EXPECT_EQ(0, (test_submit_and_wait_on_group<just_call_submit, policy_pointer_t>(u_ptrs, f_ptrs, deref_op)), "");
        EXPECT_EQ(0, (test_submit_and_wait_on_group<call_select_before_submit, policy_pointer_t>(u_ptrs, f_ptrs, deref_op)), "");

        bProcessed = true;
    }
#endif // TEST_DYNAMIC_SELECTION_AVAILABLE

    return TestUtils::done(bProcessed);
}
