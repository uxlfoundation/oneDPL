// -*- C++ -*-
//===---------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _ONEDPL_TEST_OFFSET_UTILS_H
#define _ONEDPL_TEST_OFFSET_UTILS_H

#include <thread>
#include <chrono>
#include <random>
#include <algorithm>
#include <iostream>

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
test_submit_and_wait_on_group(UniverseContainer u, ResourceFunction&& f, size_t offset = 0)
{
    using my_policy_t = Policy;
    my_policy_t p{u, offset};

    int N = 100;
    std::atomic<int> ecount = 0;
    bool pass = true;
    if constexpr (call_select_before_submit)
    {
        for (int i = 1; i <= N; i++)
        {
            auto test_resource = f(i, offset);
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
            auto test_resource = f(i, offset);
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
test_submit_and_wait_on_event(UniverseContainer u, ResourceFunction&& f, size_t offset = 0)
{
    using my_policy_t = Policy;
    my_policy_t p{u, offset};

    const int N = 100;
    bool pass = true;

    std::atomic<int> ecount = 0;

    if constexpr (call_select_before_submit)
    {
        for (int i = 1; i <= N; ++i)
        {
            auto test_resource = f(i, offset);
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
            auto test_resource = f(i, offset);
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
test_submit_and_wait(UniverseContainer u, ResourceFunction&& f, size_t offset = 0)
{
    using my_policy_t = Policy;
    my_policy_t p{u, offset};

    const int N = 100;
    std::atomic<int> ecount = 0;
    bool pass = true;

    if constexpr (call_select_before_submit)
    {
        for (int i = 1; i <= N; ++i)
        {
            auto test_resource = f(i, offset);
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
            auto test_resource = f(i, offset);
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

#endif /* _ONEDPL_TEST_OFFSET_UTILS_H */
