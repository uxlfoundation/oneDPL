// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "support/test_round_robin_policy_parallel_utils.h"

#if TEST_DYNAMIC_SELECTION_AVAILABLE
static inline void
build_universe(std::vector<sycl::queue>& u, std::unordered_map<sycl::queue, int>& map)
{
    int i=0;
    try
    {
        auto device_default = sycl::device(sycl::default_selector_v);
        sycl::queue default_queue(device_default);
        u.push_back(default_queue);
        map[default_queue] = i++;
    }
    catch (const sycl::exception&)
    {
        std::cout << "SKIPPED: Unable to run with default_selector\n";
    }

    try
    {
        auto device_gpu = sycl::device(sycl::gpu_selector_v);
        sycl::queue gpu_queue(device_gpu);
        u.push_back(gpu_queue);
        map[gpu_queue] = i++;
    }
    catch (const sycl::exception&)
    {
        std::cout << "SKIPPED: Unable to run with gpu_selector\n";
    }
    try
    {
        auto device_cpu = sycl::device(sycl::cpu_selector_v);
        sycl::queue cpu_queue(device_cpu);
        u.push_back(cpu_queue);
        map[cpu_queue] = i++;
    }
    catch (const sycl::exception&)
    {
        std::cout << "SKIPPED: Unable to run with cpu_selector\n";
    }
}
#endif // TEST_DYNAMIC_SELECTION_AVAILABLE

int
main()
{
    bool bProcessed = false;

#if TEST_DYNAMIC_SELECTION_AVAILABLE
    using policy_t = oneapi::dpl::experimental::round_robin_policy<oneapi::dpl::experimental::sycl_backend>;
    std::vector<sycl::queue> u;
    std::unordered_map<sycl::queue, int> map;
    build_universe(u, map);
    int constexpr count = 50;
    int actual;
    if (!u.empty())
    {
        constexpr bool just_call_submit = false;
        constexpr bool call_select_before_submit = true;

        actual = test_submit_and_wait_on_event<call_select_before_submit, policy_t>(u, map, build_result(u.size(), count));
        actual = test_submit_and_wait_on_group<call_select_before_submit, policy_t>(u, map, build_result(u.size(), count));
        actual = test_submit_and_wait<call_select_before_submit, policy_t>(u, map, build_result(u.size(), count));
        //Just call submit
        actual = test_submit_and_wait_on_event<just_call_submit, policy_t>(u, map, build_result(u.size(), count));
        actual = test_submit_and_wait_on_group<just_call_submit, policy_t>(u, map, build_result(u.size(), count));
        actual = test_submit_and_wait<just_call_submit, policy_t>(u, map, build_result(u.size(), count));

        bProcessed = true;
    }
#endif //TEST_DYNAMIC_SELECTION_AVAILABLE
    return TestUtils::done(bProcessed);
}
