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
#include "oneapi/dpl/functional"
#include "support/test_dynamic_selection_utils.h"

template <typename Policy, typename ResourceContainer, typename FunctionType, typename... Args>
int run_round_robin_policy_tests(const ResourceContainer& resources, const FunctionType& f, Args&&... args)
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
        auto n = u.size();
        std::cout << "UNIVERSE SIZE " << n << std::endl;

        // Test with direct sycl::queue resources
        using policy_t = oneapi::dpl::experimental::round_robin_policy<sycl::queue, oneapi::dpl::identity, oneapi::dpl::experimental::default_backend<sycl::queue>>;
        auto f = [u, n](int i) { return u[(i - 1) % n]; };

        std::cout << "\nRunning round robin tests for sycl::queue ...\n";
        EXPECT_EQ(0, (run_round_robin_policy_tests<policy_t>(u, f)), "");

        // Test with sycl::queue* resources and dereference adapter
        auto deref_op = [](auto pointer){return *pointer;};
        using policy_pointer_t = oneapi::dpl::experimental::round_robin_policy<sycl::queue*, decltype(deref_op), oneapi::dpl::experimental::default_backend<sycl::queue*, decltype(deref_op)>>;
        
        std::vector<sycl::queue*> u_ptrs;
        u_ptrs.reserve(u.size());
        for (auto& e: u)
        {
            u_ptrs.push_back(&e);
        }
        auto f_ptrs = [u_ptrs, n](int i) { return u_ptrs[(i - 1) % n]; };

        std::cout << "\nRunning round robin tests for sycl::queue* ...\n";
        EXPECT_EQ(0, (run_round_robin_policy_tests<policy_pointer_t>(u_ptrs, f_ptrs, deref_op)), "");

        bProcessed = true;
    }
#endif // TEST_DYNAMIC_SELECTION_AVAILABLE

    return TestUtils::done(bProcessed);
}
