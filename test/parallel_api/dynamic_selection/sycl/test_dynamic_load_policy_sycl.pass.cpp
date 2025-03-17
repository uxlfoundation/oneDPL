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
#include <iostream>
#include "support/test_dynamic_load_utils.h"
#include "support/utils.h"
#if TEST_DYNAMIC_SELECTION_AVAILABLE

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

int
main()
{
    bool bProcessed = false;

#if TEST_DYNAMIC_SELECTION_AVAILABLE
#if !ONEDPL_FPGA_DEVICE || !ONEDPL_FPGA_EMULATOR
    using policy_t = oneapi::dpl::experimental::dynamic_load_policy<sycl::queue, oneapi::dpl::experimental::default_backend<sycl::queue>>;
    std::vector<sycl::queue> u;
    build_dl_universe(u);

    auto n = u.size();

    //If building the universe is not a success, return
    if (n != 0)
    {
        // should be similar to round_robin when waiting on policy
        auto f = [u, n](int i) { return u[i % u.size()]; };

        auto f2 = [u, n](int i) { return u[0]; };
        // should always pick first when waiting on sync in each iteration

        constexpr bool just_call_submit = false;
        constexpr bool call_select_before_submit = true;

        auto actual = test_dl_initialization(u);
        actual = test_select<policy_t, decltype(u), decltype(f2)&, false>(u, f2);
        actual = test_submit_and_wait_on_event<just_call_submit, policy_t>(u, f2);
        actual = test_submit_and_wait_on_event<call_select_before_submit, policy_t>(u, f2);
        actual = test_submit_and_wait<just_call_submit, policy_t>(u, f2);
        actual = test_submit_and_wait<call_select_before_submit, policy_t>(u, f2);
        actual = test_submit_and_wait_on_group<just_call_submit, policy_t>(u, f);
        actual = test_submit_and_wait_on_group<call_select_before_submit, policy_t>(u, f);

        bProcessed = true;
    }
#endif // Devices available are CPU and GPU
#endif // TEST_DYNAMIC_SELECTION_AVAILABLE

    return TestUtils::done(bProcessed);
}
