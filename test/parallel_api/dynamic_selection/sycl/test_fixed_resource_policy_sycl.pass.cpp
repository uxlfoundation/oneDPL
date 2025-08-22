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
#include "oneapi/dpl/functional"
#include "support/test_dynamic_selection_utils.h"
#include "support/utils.h"

template <typename Policy, typename ResourceContainer, typename FunctionType, typename... Args>
int run_fixed_resource_policy_tests(const ResourceContainer& resources, const FunctionType& f, Args&&... args)
{
    int result = 0;
    constexpr bool just_call_submit = false;
    constexpr bool call_select_before_submit = true;
    
    result += test_initialization<Policy, typename ResourceContainer::value_type>(resources, std::forward<Args>(args)...);
    result += test_select<Policy, ResourceContainer, const FunctionType&, false>(resources, f, std::forward<Args>(args)...);
    result += test_submit_and_wait_on_event<just_call_submit, Policy>(resources, f, std::forward<Args>(args)...);
    result += test_submit_and_wait_on_event<call_select_before_submit, Policy>(resources, f, std::forward<Args>(args)...);
    result += test_submit_and_wait<just_call_submit, Policy>(resources, f, std::forward<Args>(args)...);
    result += test_submit_and_wait<call_select_before_submit, Policy>(resources, f, std::forward<Args>(args)...);
    result += test_submit_and_wait_on_group<just_call_submit, Policy>(resources, f, std::forward<Args>(args)...);
    result += test_submit_and_wait_on_group<call_select_before_submit, Policy>(resources, f, std::forward<Args>(args)...);
    
    return result;
}

int
main()
{
    bool bProcessed = false;

#if TEST_DYNAMIC_SELECTION_AVAILABLE
    std::vector<sycl::queue> u;
    build_universe(u);
    if (!u.empty())
    {
        // Test with direct sycl::queue resources
        using policy_t = oneapi::dpl::experimental::fixed_resource_policy<sycl::queue, oneapi::dpl::identity, oneapi::dpl::experimental::default_backend<sycl::queue, oneapi::dpl::identity>>;
        auto f = [u](int, int offset = 0) { return u[offset]; };
        
        std::cout<<"\nRunning tests for sycl::queue ...\n";
        EXPECT_EQ(0, (run_fixed_resource_policy_tests<policy_t>(u, f)), "");
        
        // Test with sycl::queue* resources and dereference adapter
        auto deref_op = [](auto pointer){return *pointer;};
        using policy_pointer_t = oneapi::dpl::experimental::fixed_resource_policy<sycl::queue*, decltype(deref_op), oneapi::dpl::experimental::default_backend<sycl::queue*, decltype(deref_op)>>;
        
        std::vector<sycl::queue*> u_ptrs;
        u_ptrs.reserve(u.size());
        for (auto& e: u)
        {
            u_ptrs.push_back(&e);
        }
        auto f_ptrs = [u_ptrs](int, int offset = 0) { return u_ptrs[offset]; };

        std::cout<<"\nRunning tests for sycl::queue* ...\n";
        EXPECT_EQ(0, (run_fixed_resource_policy_tests<policy_pointer_t>(u_ptrs, f_ptrs, deref_op)), "");

        bProcessed = true;
    }
#endif // TEST_DYNAMIC_SELECTION_AVAILABLE

    return TestUtils::done(bProcessed);
}
