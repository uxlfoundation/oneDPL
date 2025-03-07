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
#include <thread>
#include "support/test_dynamic_selection_utils.h"
#include "support/inline_backend.h"
#include "support/utils.h"

template <typename Policy, typename UniverseContainer, bool do_select = false>
int
test_auto_submit(UniverseContainer u, int best_resource)
{
    using my_policy_t = Policy;

    my_policy_t p{u};
    auto n_samples = u.size();

    const int N = 12;
    std::atomic<int> ecount = 0;
    bool pass = true;

    for (int i = 1; i <= N; ++i)
    {
        // we can capture all by reference
        // the inline_scheduler reports timings in submit
        // so we can treat this just like submit_and_wait
        if constexpr (do_select)
        {
            auto f = [&](typename oneapi::dpl::experimental::policy_traits<Policy>::resource_type e) {
                std::this_thread::sleep_for(std::chrono::milliseconds(e));
                if (i <= 2 * n_samples)
                {
                    // we should be round-robining through the resources
                    if (e != u[(i - 1) % n_samples])
                    {
                        pass = false;
                    }
                }
                else
                {
                    if (e != best_resource)
                    {
                        pass = false;
                    }
                }
                ecount += i;
                return typename oneapi::dpl::experimental::policy_traits<Policy>::wait_type{};
            };
            auto s = oneapi::dpl::experimental::select(p, f);
            oneapi::dpl::experimental::submit(s, f);
        }
        else
        {
            oneapi::dpl::experimental::submit(
                p, [&](typename oneapi::dpl::experimental::policy_traits<Policy>::resource_type e) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(e));
                    if (i <= 2 * n_samples)
                    {
                        // we should be round-robining through the resources
                        if (e != u[(i - 1) % n_samples])
                        {
                            pass = false;
                        }
                    }
                    else
                    {
                        if (e != best_resource)
                        {
                            pass = false;
                        }
                    }
                    ecount += i;
                    return typename oneapi::dpl::experimental::policy_traits<Policy>::wait_type{};
                });
        }
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
    std::cout << "submit: OK\n";
    return 0;
}

template <typename Policy, typename UniverseContainer, bool do_select = false>
int
test_auto_submit_wait_on_event(UniverseContainer u, int best_resource)
{
    using my_policy_t = Policy;

    my_policy_t p{u};
    auto n_samples = u.size();

    const int N = 12;
    std::atomic<int> ecount = 0;
    bool pass = true;

    for (int i = 1; i <= N; ++i)
    {
        // we can capture all by reference
        // the inline_scheduler reports timings in submit
        // We wait but it should return immediately, since inline
        // scheduler does the work "inline".
        // The unwrapped wait type should be equal to the resource
        int e_val = -1;
        if constexpr (do_select)
        {
            auto f = [&](typename oneapi::dpl::experimental::policy_traits<Policy>::resource_type e) {
                std::this_thread::sleep_for(std::chrono::milliseconds(e));
                if (i <= 2 * n_samples)
                {
                    // we should be round-robining through the resources
                    if (e != u[(i - 1) % n_samples])
                    {
                        pass = false;
                    }
                }
                else
                {
                    if (e != best_resource)
                    {
                        pass = false;
                    }
                }
                ecount += i;
                return e; // we return the device we were given
            };
            auto s = oneapi::dpl::experimental::select(p, f);
            auto e = oneapi::dpl::experimental::submit(s, f);
            oneapi::dpl::experimental::wait(e);
            e_val = oneapi::dpl::experimental::unwrap(e);
        }
        else
        {
            auto e = oneapi::dpl::experimental::submit(
                p, [&](typename oneapi::dpl::experimental::policy_traits<Policy>::resource_type e) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(e));
                    if (i <= 2 * n_samples)
                    {
                        // we should be round-robining through the resources
                        if (e != u[(i - 1) % n_samples])
                        {
                            pass = false;
                        }
                    }
                    else
                    {
                        if (e != best_resource)
                        {
                            pass = false;
                        }
                    }
                    ecount += i;
                    return e; // we return the device we were given
                });
            oneapi::dpl::experimental::wait(e);
            e_val = oneapi::dpl::experimental::unwrap(e);
        }

        if (i <= 2 * n_samples && e_val != u[(i - 1) % n_samples])
        {
            std::cout << "ERROR: wrong value in unwrapped wait_type when sampling " << i << ": " << e_val
                      << " != " << u[i - 1] << "\n";
            return 1;
        }
        else if (i > 2 * n_samples && e_val != best_resource)
        {
            std::cout << "ERROR: wrong value in unwrapped wait_type " << i << ": " << e_val << " != " << best_resource
                      << "\n";
            return 1;
        }

        int count = ecount.load();
        if (count != i * (i + 1) / 2)
        {
            std::cout << "ERROR: scheduler did not execute all tasks exactly once\n";
            return 1;
        }
    }
    EXPECT_TRUE(pass, "ERROR: did not select expected resources\n");
    std::cout << "submit and wait on event: OK\n";
    return 0;
}

template <typename Policy, typename UniverseContainer, bool do_select = false>
int
test_auto_submit_wait_on_group(UniverseContainer u, int best_resource)
{
    using my_policy_t = Policy;

    my_policy_t p{u};
    auto n_samples = u.size();

    const int N = 12;
    std::atomic<int> ecount = 0;
    bool pass = true;

    for (int i = 1; i <= N; ++i)
    {
        // we can capture all by reference, since it should wait, no concurrency
        if constexpr (do_select)
        {
            auto f = [&](typename oneapi::dpl::experimental::policy_traits<Policy>::resource_type e) {
                std::this_thread::sleep_for(std::chrono::milliseconds(e));
                if (i <= 2 * n_samples)
                {
                    // we should be round-robining through the resources
                    if (e != u[(i - 1) % n_samples])
                    {
                        pass = false;
                    }
                }
                else
                {
                    if (e != best_resource)
                    {
                        pass = false;
                    }
                }
                ecount += i;
                return typename oneapi::dpl::experimental::policy_traits<Policy>::wait_type{};
            };
            auto s = oneapi::dpl::experimental::select(p, f);
            oneapi::dpl::experimental::submit(s, f);
        }
        else
        {
            oneapi::dpl::experimental::submit(
                p, [&](typename oneapi::dpl::experimental::policy_traits<Policy>::resource_type e) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(e));
                    if (i <= 2 * n_samples)
                    {
                        // we should be round-robining through the resources
                        if (e != u[(i - 1) % n_samples])
                        {
                            pass = false;
                        }
                    }
                    else
                    {
                        if (e != best_resource)
                        {
                            pass = false;
                        }
                    }
                    ecount += i;
                    return typename oneapi::dpl::experimental::policy_traits<Policy>::wait_type{};
                });
        }
        int count = ecount.load();
        EXPECT_EQ(i * (i + 1) / 2, count, "ERROR: scheduler did not execute all tasks exactly once\n");
    }
    // this has no effect for inline_scheduler, so nothing to test other than the call
    // doesn't fail
    oneapi::dpl::experimental::wait(p.get_submission_group());

    EXPECT_TRUE(pass, "ERROR: did not select expected resources\n");
    std::cout << "submit_wait_on_group: OK\n";
    return 0;
}

template <typename Policy, typename UniverseContainer, bool do_select = false>
int
test_auto_submit_and_wait(UniverseContainer u, int best_resource)
{
    using my_policy_t = Policy;

    my_policy_t p{u};
    auto n_samples = u.size();

    const int N = 12;
    std::atomic<int> ecount = 0;
    bool pass = true;

    for (int i = 1; i <= N; ++i)
    {
        // we can capture all by reference, since it should wait, no concurrency
        if constexpr (do_select)
        {
            auto f = [&](typename oneapi::dpl::experimental::policy_traits<Policy>::resource_type e) {
                std::this_thread::sleep_for(std::chrono::milliseconds(e));
                if (i <= 2 * n_samples)
                {
                    // we should be round-robining through the resources
                    if (e != u[(i - 1) % n_samples])
                    {
                        pass = false;
                    }
                }
                else
                {
                    if (e != best_resource)
                    {
                        pass = false;
                    }
                }
                ecount += i;
                return typename oneapi::dpl::experimental::policy_traits<Policy>::wait_type{};
            };
            auto s = oneapi::dpl::experimental::select(p, f);
            oneapi::dpl::experimental::submit_and_wait(s, f);
        }
        else
        {
            oneapi::dpl::experimental::submit_and_wait(
                p, [&](typename oneapi::dpl::experimental::policy_traits<Policy>::resource_type e) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(e));
                    if (i <= 2 * n_samples)
                    {
                        // we should be round-robining through the resources
                        if (e != u[(i - 1) % n_samples])
                        {
                            pass = false;
                        }
                    }
                    else
                    {
                        if (e != best_resource)
                        {
                            pass = false;
                        }
                    }
                    ecount += i;
                    return typename oneapi::dpl::experimental::policy_traits<Policy>::wait_type{};
                });
        }
        int count = ecount.load();
        EXPECT_EQ(i * (i + 1) / 2, count, "ERROR: scheduler did not execute all tasks exactly once\n");
    }
    EXPECT_TRUE(pass, "ERROR: did not select expected resources\n");
    std::cout << "submit_and_wait: OK\n";
    return 0;
}

template <typename Policy>
void
run_tests(std::vector<int> u, int best_resource)
{
    using policy_t = Policy;

    // We know there are 4 resources. We know that without submitting,
    // that auto_tune acts like round-robin until sampling is done.
    // So the first 4 are round-robin and then afterwards, with no better
    // info, the first resource is used.
    auto f = [u](int i) {
        if (i <= 8)
            return u[(i - 1) % 4];
        else
            return u[0];
    };

    auto actual = test_initialization<policy_t>(u);
    actual = test_select<policy_t, decltype(u), const decltype(f)&, true>(u, f);
    actual = test_auto_submit<policy_t>(u, best_resource);
    actual = test_auto_submit_wait_on_event<policy_t>(u, best_resource);
    actual = test_auto_submit_wait_on_group<policy_t>(u, best_resource);
    actual = test_auto_submit_and_wait<policy_t>(u, best_resource);
    // now select then submits
    actual = test_auto_submit<policy_t, decltype(u), true>(u, best_resource);
    actual = test_auto_submit_wait_on_event<policy_t, decltype(u), true>(u, best_resource);
    actual = test_auto_submit_wait_on_group<policy_t, decltype(u), true>(u, best_resource);
    actual = test_auto_submit_and_wait<policy_t, decltype(u), true>(u, best_resource);
}

int
main()
{
    using policy_t = oneapi::dpl::experimental::auto_tune_policy<TestUtils::int_inline_backend_t>;
    std::vector<int> first_resources = {1, 100, 100, 100};
    std::vector<int> second_resources = {100, 1, 100, 100};
    std::vector<int> third_resources = {100, 100, 1, 100};
    std::vector<int> fourth_resources = {100, 100, 100, 1};

    run_tests<policy_t>(first_resources, 1);
    run_tests<policy_t>(second_resources, 1);
    run_tests<policy_t>(third_resources, 1);
    run_tests<policy_t>(fourth_resources, 1);

    return TestUtils::done();
}
