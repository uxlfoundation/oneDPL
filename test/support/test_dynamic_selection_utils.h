// -*- C++ -*-
//===---------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _ONEDPL_TEST_DYNAMIC_SELECTION_UTILS_H
#define _ONEDPL_TEST_DYNAMIC_SELECTION_UTILS_H

#include <thread>
#include <chrono>
#include <random>
#include <algorithm>
#include <iostream>
#include "support/utils_dynamic_selection.h"

#if TEST_DYNAMIC_SELECTION_AVAILABLE
#include "utils_sycl_defs.h"

static inline void
build_universe(std::vector<sycl::queue>& u)
{
    try
    {
        auto device_default = sycl::device(sycl::default_selector_v);
        sycl::queue default_queue(device_default);
        u.push_back(default_queue);
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
    }
    catch (const sycl::exception&)
    {
        std::cout << "SKIPPED: Unable to run with cpu_selector\n";
    }
}

#endif // TEST_DYNAMIC_SELECTION_AVAILABLE
template <typename Policy, typename T, typename... Args>
int
test_initialization(const std::vector<T>& u, Args... args)
{
    // initialize
    using my_policy_t = Policy;
    my_policy_t p{u, args...};
    auto u2 = oneapi::dpl::experimental::get_resources(p);
    if (!std::equal(std::begin(u2), std::end(u2), std::begin(u)))
    {
        std::cout << "ERROR: provided resources and queried resources are not equal\n";
        return 1;
    }

    // deferred initialization
    my_policy_t p2{oneapi::dpl::experimental::deferred_initialization};
    try
    {
        auto u3 = oneapi::dpl::experimental::get_resources(p2);
        if (!u3.empty())
        {
            std::cout << "ERROR: deferred initialization not respected\n";
            return 1;
        }
    }
    catch (...)
    {
    }
    p2.initialize(u, args...);
    auto u3 = oneapi::dpl::experimental::get_resources(p);
    if (!std::equal(std::begin(u3), std::end(u3), std::begin(u)))
    {
        std::cout << "ERROR: reported resources and queried resources are not equal after deferred initialization\n";
        return 1;
    }

    std::cout << "initialization: OK\n" << std::flush;
    return 0;
}

template <typename Policy, typename Backend, typename ResourceAdapter, typename... Args>
int
test_default_universe_initialization(ResourceAdapter, [[maybe_unused]] Args&&... args)
{
    // Default universe initialization only works with identity adapter
    // Check if Policy has a resource type of a queue or if it has a custom adapter
    if constexpr (!std::is_same_v<ResourceAdapter, oneapi::dpl::identity>)
    {
        std::cout << "default universe initialization: SKIPPED (custom adapter)\n" << std::flush;
        return 0;
    }
    else
    {
        // Test default universe initialization (no resource vector provided)
        Policy p{oneapi::dpl::experimental::deferred_initialization};
        p.initialize();

        // Verify that we got some resources from default initialization
        auto u = oneapi::dpl::experimental::get_resources(p);
        if (u.empty())
        {
            std::cout << "ERROR: default universe initialization resulted in empty resources\n";
            return 1;
        }

        // Test that we can actually submit to the default universe
        bool executed = false;
        oneapi::dpl::experimental::submit_and_wait(p, [&executed](auto e) {
            executed = true;
            if constexpr (std::is_same_v<typename oneapi::dpl::experimental::policy_traits<Policy>::resource_type, int>)
                return e;
            else
                return typename TestUtils::get_wait_type<Backend>::type{};
        });

        if (!executed)
        {
            std::cout << "ERROR: default universe initialization did not execute task\n";
            return 1;
        }

        std::cout << "default universe initialization: OK\n" << std::flush;
        return 0;
    }
}

template <typename Policy, typename Backend, typename UniverseContainer, typename ResourceFunction, typename... Args>
int
test_submit_and_wait_on_group(UniverseContainer u, ResourceFunction&& f, Args... args)
{
    using my_policy_t = Policy;
    my_policy_t p{u, args...};

    int N = 100;
    std::atomic<int> ecount = 0;
    bool pass = true;
    for (int i = 1; i <= N; ++i)
    {
        auto test_resource = f(i);
        oneapi::dpl::experimental::submit(
            p, [&pass, &ecount, test_resource,
                i](typename oneapi::dpl::experimental::policy_traits<Policy>::resource_type e) {
                if (e != test_resource)
                {
                    pass = false;
                }
                ecount += i;
                if constexpr (std::is_same_v<typename oneapi::dpl::experimental::policy_traits<Policy>::resource_type,
                                             int>)
                    return e;
                else
                    return typename TestUtils::get_wait_type<Backend>::type{};
            });
    }
    oneapi::dpl::experimental::wait(p.get_submission_group());
    if (!pass)
    {
        std::cout << "ERROR: did not select expected resources\n";
        return 1;
    }
    std::cout << "submit_and_wait_on_group: OK\n";
    return 0;
}

template <typename Policy, typename Backend, typename UniverseContainer, typename ResourceFunction, typename... Args>
int
test_submit_and_wait_on_event(UniverseContainer u, ResourceFunction&& f, Args... args)
{
    using my_policy_t = Policy;
    my_policy_t p{u, args...};

    const int N = 100;
    bool pass = true;

    std::atomic<int> ecount = 0;

    for (int i = 1; i <= N; ++i)
    {
        auto test_resource = f(i);
        auto w = oneapi::dpl::experimental::submit(
            p, [&pass, test_resource, &ecount,
                i](typename oneapi::dpl::experimental::policy_traits<Policy>::resource_type e) {
                if (e != test_resource)
                {
                    pass = false;
                }
                ecount += i;
                if constexpr (std::is_same_v<typename oneapi::dpl::experimental::policy_traits<Policy>::resource_type,
                                             int>)
                    return e;
                else
                    return typename TestUtils::get_wait_type<Backend>::type{};
            });
        oneapi::dpl::experimental::wait(w);
        int count = ecount.load();
        if (count != i * (i + 1) / 2)
        {
            std::cout << "ERROR: scheduler did not execute all tasks exactly once\n";
            return 1;
        }
    }
    if (!pass)
    {
        std::cout << "ERROR: did not select expected resources\n";
        return 1;
    }
    std::cout << "submit_and_wait_on_sync: OK\n";
    return 0;
}

template <typename Policy, typename Backend, typename UniverseContainer, typename ResourceFunction, typename... Args>
int
test_submit_and_wait(UniverseContainer u, ResourceFunction&& f, Args... args)
{
    using my_policy_t = Policy;
    my_policy_t p{u, args...};

    const int N = 100;
    std::atomic<int> ecount = 0;
    bool pass = true;

    for (int i = 1; i <= N; ++i)
    {
        auto test_resource = f(i);
        oneapi::dpl::experimental::submit_and_wait(
            p, [&pass, &ecount, test_resource,
                i](typename oneapi::dpl::experimental::policy_traits<Policy>::resource_type e) {
                if (e != test_resource)
                {
                    pass = false;
                }
                ecount += i;
                if constexpr (std::is_same_v<typename oneapi::dpl::experimental::policy_traits<Policy>::resource_type,
                                             int>)
                    return e;
                else
                    return typename TestUtils::get_wait_type<Backend>::type{};
            });
        int count = ecount.load();
        if (count != i * (i + 1) / 2)
        {
            std::cout << "ERROR: scheduler did not execute all tasks exactly once\n";
            return 1;
        }
    }
    if (!pass)
    {
        std::cout << "ERROR: did not select expected resources\n";
        return 1;
    }
    std::cout << "submit_and_wait: OK\n";
    return 0;
}


#endif /* _ONEDPL_TEST_DYNAMIC_SELECTION_UTILS_H */
