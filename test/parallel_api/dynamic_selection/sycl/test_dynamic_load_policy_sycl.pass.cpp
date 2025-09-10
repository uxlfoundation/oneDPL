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
#include <iostream>
#include "support/test_dynamic_load_utils.h"
#include "support/utils.h"
#if TEST_DYNAMIC_SELECTION_AVAILABLE

template <typename CustomName, typename Policy, typename ResourceContainer, typename FunctionType, typename FunctionType2, typename... Args>
int run_dynamic_load_policy_tests(const ResourceContainer& resources, const FunctionType& f, const FunctionType2& f2, Args&&... args)
{
    int result = 0;
    constexpr bool just_call_submit = false;
    constexpr bool call_select_before_submit = true;

    result += test_dl_initialization<Policy, ResourceContainer>(resources, std::forward<Args>(args)...);
    result += test_select<Policy, ResourceContainer, const FunctionType2&, false>(resources, f2, std::forward<Args>(args)...);
    result += test_submit_and_wait_on_event<just_call_submit, Policy>(resources, f2, std::forward<Args>(args)...);
    result += test_submit_and_wait_on_event<call_select_before_submit, Policy>(resources, f2, std::forward<Args>(args)...);
    result += test_submit_and_wait<just_call_submit, Policy>(resources, f2, std::forward<Args>(args)...);
    result += test_submit_and_wait<call_select_before_submit, Policy>(resources, f2, std::forward<Args>(args)...);
    result += test_submit_and_wait_on_group<just_call_submit, TestUtils::unique_kernel_name<CustomName, 0>, Policy>(resources, f, std::forward<Args>(args)...);
    result += test_submit_and_wait_on_group<call_select_before_submit, TestUtils::unique_kernel_name<CustomName, 1>, Policy>(resources, f, std::forward<Args>(args)...);

    return result;
}

static inline void
build_dl_universe(std::vector<sycl::queue>& u)
{
    try
    {
        auto device_cpu1 = sycl::device(sycl::cpu_selector_v);
        sycl::queue cpu1_queue(device_cpu1);
        u.push_back(cpu1_queue);
    }
    catch (const sycl::exception&)
    {
        std::cout << "SKIPPED: Unable to run with cpu_selector\n";
    }
    try
    {
        auto device_cpu2 = sycl::device(sycl::cpu_selector_v);
        sycl::queue cpu2_queue(device_cpu2);
        u.push_back(cpu2_queue);
    }
    catch (const sycl::exception&)
    {
        std::cout << "SKIPPED: Unable to run with cpu_selector\n";
    }
}
#endif

struct queue_load;
struct queue_ptr_load;


int
main()
{
    bool bProcessed = false;

#if TEST_DYNAMIC_SELECTION_AVAILABLE
#if !ONEDPL_FPGA_DEVICE || !ONEDPL_FPGA_EMULATOR
    std::vector<sycl::queue> u;
    build_dl_universe(u);

    auto n = u.size();

    //If building the universe is not a success, return
    if (n != 0)
    {
        // Test with direct sycl::queue resources
        using policy_t = oneapi::dpl::experimental::dynamic_load_policy<sycl::queue, oneapi::dpl::identity, oneapi::dpl::experimental::default_backend<sycl::queue>>;
        
        // should be similar to round_robin when waiting on policy
        auto f = [u](int i) { return u[i % u.size()]; };
        auto f2 = [u](int) { return u[0]; };
        // should always pick first when waiting on sync in each iteration

        std::cout << "\nRunning dynamic load tests for sycl::queue ...\n";
        EXPECT_EQ(0, (run_dynamic_load_policy_tests<queue_load, policy_t>(u, f, f2)), "");

        // Test with sycl::queue* resources and dereference adapter
        auto deref_op = [](auto pointer){return *pointer;};
        using policy_pointer_t = oneapi::dpl::experimental::dynamic_load_policy<sycl::queue*, decltype(deref_op), oneapi::dpl::experimental::default_backend<sycl::queue*, decltype(deref_op)>>;
        
        std::vector<sycl::queue*> u_ptrs;
        u_ptrs.reserve(u.size());
        for (auto& e: u)
        {
            u_ptrs.push_back(&e);
        }
        auto f_ptrs = [u_ptrs](int i) { return u_ptrs[i % u_ptrs.size()]; };
        auto f2_ptrs = [u_ptrs](int) { return u_ptrs[0]; };

        std::cout << "\nRunning dynamic load tests for sycl::queue* ...\n";
        EXPECT_EQ(0, (run_dynamic_load_policy_tests<queue_ptr_load, policy_pointer_t>(u_ptrs, f_ptrs, f2_ptrs, deref_op)), "");

        bProcessed = true;
    }
#endif // Devices available are CPU and GPU
#endif // TEST_DYNAMIC_SELECTION_AVAILABLE

    return TestUtils::done(bProcessed);
}
