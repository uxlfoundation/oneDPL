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
#include "support/test_dynamic_selection_utils.h"
#include "support/utils.h"
#if TEST_DYNAMIC_SELECTION_AVAILABLE

template <typename CustomName, typename Policy, typename Backend, typename ResourceContainer, typename ResourceAdapter,
          typename FunctionType, typename FunctionType2, typename... Args>
int
run_dynamic_load_policy_tests(const ResourceContainer& resources, const FunctionType& f, const FunctionType2& f2,
                              ResourceAdapter adapter, Args&&... args)
{
    int result = 0;

    result += test_dl_initialization<Policy, ResourceContainer>(resources, adapter, std::forward<Args>(args)...);
    result += test_default_universe_initialization<Policy, Backend>(adapter, std::forward<Args>(args)...);
    result += test_submit_and_wait_on_event<Policy, Backend>(resources, f2, adapter, std::forward<Args>(args)...);
    result += test_submit_and_wait_on_event<Policy, Backend>(resources, f2, adapter, std::forward<Args>(args)...);
    result += test_submit_and_wait<Policy, Backend>(resources, f2, adapter, std::forward<Args>(args)...);
    result += test_submit_and_wait<Policy, Backend>(resources, f2, adapter, std::forward<Args>(args)...);
    result += test_dl_submit_and_wait_on_group<TestUtils::unique_kernel_name<CustomName, 0>, Policy>(
        resources, f, adapter, std::forward<Args>(args)...);
    result += test_dl_submit_and_wait_on_group<TestUtils::unique_kernel_name<CustomName, 1>, Policy>(
        resources, f, adapter, std::forward<Args>(args)...);

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

    try
    {
#if TEST_DYNAMIC_SELECTION_AVAILABLE
#if !ONEDPL_FPGA_DEVICE || !ONEDPL_FPGA_EMULATOR
        std::vector<sycl::queue> u;
        build_dl_universe(u);

        auto n = u.size();

        //If building the universe is not a success, return
        if (n != 0)
        {
            bProcessed = true;

            // Test with direct sycl::queue resources
            using policy_t =
                oneapi::dpl::experimental::dynamic_load_policy<sycl::queue, oneapi::dpl::identity,
                                                               oneapi::dpl::experimental::default_backend<sycl::queue>>;

            // should be similar to round_robin when waiting on policy
            auto f = [u](int i) { return u[i % u.size()]; };
            auto f2 = [u](int) { return u[0]; };
            // should always pick first when waiting on sync in each iteration

            std::cout << "\nRunning dynamic load tests for sycl::queue ...\n";
            EXPECT_EQ(0, (run_dynamic_load_policy_tests<queue_load, policy_t, oneapi::dpl::experimental::default_backend<sycl::queue>>(u, f, f2, oneapi::dpl::identity{})), "");

            // Test with sycl::queue* resources and dereference adapter
            auto deref_op = [](auto pointer) { return *pointer; };
            using policy_pointer_t = oneapi::dpl::experimental::dynamic_load_policy<
                sycl::queue*, decltype(deref_op),
                oneapi::dpl::experimental::default_backend<sycl::queue*, decltype(deref_op)>>;

            std::vector<sycl::queue*> u_ptrs;
            u_ptrs.reserve(u.size());
            for (auto& e : u)
            {
                u_ptrs.push_back(&e);
            }
            auto f_ptrs = [u_ptrs](int i) { return u_ptrs[i % u_ptrs.size()]; };
            auto f2_ptrs = [u_ptrs](int) { return u_ptrs[0]; };

            std::cout << "\nRunning dynamic load tests for sycl::queue* ...\n";
            EXPECT_EQ(0,
                (run_dynamic_load_policy_tests<queue_ptr_load, policy_pointer_t, oneapi::dpl::experimental::default_backend<sycl::queue*, decltype(deref_op)>>(u_ptrs, f_ptrs, f2_ptrs, deref_op)),
                "");

            //CTAD tests (testing policy construction without template arguments)
            //Template arguments types are deduced with CTAD
            sycl::queue q1(sycl::default_selector_v);
            sycl::queue q2(sycl::default_selector_v);
            oneapi::dpl::experimental::dynamic_load_policy p1{{q1, q2}};
            oneapi::dpl::experimental::dynamic_load_policy p2({q1, q2});

            oneapi::dpl::experimental::dynamic_load_policy p3({&q1, &q2}, deref_op);
            oneapi::dpl::experimental::dynamic_load_policy p4{{&q1, &q2}, deref_op};

        }
        else 
        {
            std::cout << "SKIPPED: No devices available to build universe (CPU or GPU required)\n";
        }
#endif // Devices available are CPU and GPU
#endif // TEST_DYNAMIC_SELECTION_AVAILABLE
    }
    catch (const std::exception& exc)
    {
        std::stringstream str;

        str << "Exception occurred";
        if (exc.what())
            str << " : " << exc.what();

        TestUtils::issue_error_message(str);
    }

    return TestUtils::done(bProcessed);
}
