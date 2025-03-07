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
#if TEST_DYNAMIC_SELECTION_AVAILABLE

namespace TestUtils
{
template <typename Op, ::std::size_t CallNumber>
struct unique_kernel_name;

template <typename Policy, int idx>
using new_kernel_name = unique_kernel_name<std::decay_t<Policy>, idx>;
} // namespace TestUtils

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
template <typename Policy, typename T>
int
test_initialization(const std::vector<T>& u)
{
    // initialize
    using my_policy_t = Policy;
    my_policy_t p{u};
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
    p2.initialize(u);
    auto u3 = oneapi::dpl::experimental::get_resources(p);
    if (!std::equal(std::begin(u3), std::end(u3), std::begin(u)))
    {
        std::cout << "ERROR: reported resources and queried resources are not equal after deferred initialization\n";
        return 1;
    }

    std::cout << "initialization: OK\n" << std::flush;
    return 0;
}

template <typename Policy, typename UniverseContainer, typename ResourceFunction, bool AutoTune = false>
int
test_select(UniverseContainer u, ResourceFunction&& f)
{
    using my_policy_t = Policy;
    my_policy_t p{u};

    const int N = 100;
    std::atomic<int> ecount = 0;
    bool pass = true;

    auto function_key = []() {};

    for (int i = 1; i <= N; ++i)
    {
        auto test_resource = f(i);
        if constexpr (AutoTune)
        {
            auto h = select(p, function_key);
            if (oneapi::dpl::experimental::unwrap(h) != test_resource)
            {
                pass = false;
            }
        }
        else
        {
            auto h = select(p);
            if (oneapi::dpl::experimental::unwrap(h) != test_resource)
            {
                pass = false;
            }
        }
        ecount += i;
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
    std::cout << "select: OK\n";
    return 0;
}

template <bool call_select_before_submit, typename Policy, typename UniverseContainer, typename ResourceFunction>
int
test_submit_and_wait_on_group(UniverseContainer u, ResourceFunction&& f, int offset = 0)
{
    using my_policy_t = Policy;
    my_policy_t p{u};

    int N = 100;
    std::atomic<int> ecount = 0;
    bool pass = true;
    if constexpr (call_select_before_submit)
    {
        for (int i = 1; i <= N; i++)
        {
            auto test_resource = f(i);
            auto func = [&](typename Policy::resource_type e) {
                if (e != test_resource)
                {
                    pass = false;
                }
                ecount += i;
                if constexpr (std::is_same_v<typename oneapi::dpl::experimental::policy_traits<Policy>::resource_type,
                                             int>)
                    return e;
                else
                    return typename oneapi::dpl::experimental::policy_traits<Policy>::wait_type{};
            };
            auto s = oneapi::dpl::experimental::select(p);
            oneapi::dpl::experimental::submit(s, func);
        }
        oneapi::dpl::experimental::wait(p.get_submission_group());
    }
    else
    {
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
                    if constexpr (std::is_same_v<
                                      typename oneapi::dpl::experimental::policy_traits<Policy>::resource_type, int>)
                        return e;
                    else
                        return typename oneapi::dpl::experimental::policy_traits<Policy>::wait_type{};
                });
        }
        oneapi::dpl::experimental::wait(p.get_submission_group());
    }
    if (!pass)
    {
        std::cout << "ERROR: did not select expected resources\n";
        return 1;
    }
    std::cout << "submit_and_wait_on_group: OK\n";
    return 0;
}

template <bool call_select_before_submit, typename Policy, typename UniverseContainer, typename ResourceFunction>
int
test_submit_and_wait_on_event(UniverseContainer u, ResourceFunction&& f, int offset = 0)
{
    using my_policy_t = Policy;
    my_policy_t p{u};

    const int N = 100;
    bool pass = true;

    std::atomic<int> ecount = 0;

    if constexpr (call_select_before_submit)
    {
        for (int i = 1; i <= N; ++i)
        {
            auto test_resource = f(i);
            auto func = [&pass, test_resource, &ecount,
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
                    return typename oneapi::dpl::experimental::policy_traits<Policy>::wait_type{};
            };
            auto s = oneapi::dpl::experimental::select(p, func);
            auto w = oneapi::dpl::experimental::submit(s, func);
            oneapi::dpl::experimental::wait(w);
            int count = ecount.load();
            if (count != i * (i + 1) / 2)
            {
                std::cout << "ERROR: scheduler did not execute all tasks exactly once\n";
                return 1;
            }
        }
    }
    else
    {
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
                    if constexpr (std::is_same_v<
                                      typename oneapi::dpl::experimental::policy_traits<Policy>::resource_type, int>)
                        return e;
                    else
                        return typename oneapi::dpl::experimental::policy_traits<Policy>::wait_type{};
                });
            oneapi::dpl::experimental::wait(w);
            int count = ecount.load();
            if (count != i * (i + 1) / 2)
            {
                std::cout << "ERROR: scheduler did not execute all tasks exactly once\n";
                return 1;
            }
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

template <bool call_select_before_submit, typename Policy, typename UniverseContainer, typename ResourceFunction>
int
test_submit_and_wait(UniverseContainer u, ResourceFunction&& f, int offset = 0)
{
    using my_policy_t = Policy;
    my_policy_t p{u};

    const int N = 100;
    std::atomic<int> ecount = 0;
    bool pass = true;

    if constexpr (call_select_before_submit)
    {
        for (int i = 1; i <= N; ++i)
        {
            auto test_resource = f(i);
            auto func = [&pass, test_resource, &ecount,
                         i](typename oneapi::dpl::experimental::policy_traits<Policy>::resource_type e) {
                if (e != test_resource)
                {
                    pass = false;
                }
                ecount += i;
                return typename oneapi::dpl::experimental::policy_traits<Policy>::wait_type{};
            };
            auto s = oneapi::dpl::experimental::select(p, func);
            oneapi::dpl::experimental::submit_and_wait(s, func);
            int count = ecount.load();
            if (count != i * (i + 1) / 2)
            {
                std::cout << "ERROR: scheduler did not execute all tasks exactly once\n";
                return 1;
            }
        }
    }
    else
    {
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
                    if constexpr (std::is_same_v<
                                      typename oneapi::dpl::experimental::policy_traits<Policy>::resource_type, int>)
                        return e;
                    else
                        return typename oneapi::dpl::experimental::policy_traits<Policy>::wait_type{};
                });
            int count = ecount.load();
            if (count != i * (i + 1) / 2)
            {
                std::cout << "ERROR: scheduler did not execute all tasks exactly once\n";
                return 1;
            }
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
