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
#include "support/utils.h"
#include "support/sycl_alloc_utils.h"

#if TEST_DYNAMIC_SELECTION_AVAILABLE

int
test_auto_initialization(const std::vector<sycl::queue>& u)
{
    // initialize
    oneapi::dpl::experimental::auto_tune_policy p{u};
    auto u2 = oneapi::dpl::experimental::get_resources(p);
    auto u2s = u2.size();
    EXPECT_TRUE(std::equal(std::begin(u2), std::end(u2), std::begin(u)),
                "ERROR: provided resources and queried resources are not equal\n");

    // deferred initialization
    oneapi::dpl::experimental::auto_tune_policy p2{oneapi::dpl::experimental::deferred_initialization};
    try
    {
        auto u3 = oneapi::dpl::experimental::get_resources(p2);
        EXPECT_TRUE(u3.empty(), "ERROR: deferred initialization not respected\n");
    }
    catch (...)
    {
    }
    p2.initialize(u);
    auto u3 = oneapi::dpl::experimental::get_resources(p);
    auto u3s = u3.size();
    EXPECT_TRUE(std::equal(std::begin(u3), std::end(u3), std::begin(u)),
                "ERROR: reported resources and queried resources are not equal after deferred initialization\n");

    std::cout << "initialization: OK\n" << std::flush;
    return 0;
}

template <bool call_select_before_submit, typename Policy, typename UniverseContainer>
int
test_auto_submit_wait_on_event(UniverseContainer u, int best_resource)
{
    using my_policy_t = Policy;

    // they are cpus so this is ok
    TestUtils::usm_data_transfer<sycl::usm::alloc::shared, double> dt_helper_v(u[0], 1000000);
    TestUtils::usm_data_transfer<sycl::usm::alloc::shared, int> dt_helper_j(u[0], 1);

    //Making v volatile so the release build does not optimize it in the for loop below
    volatile double* v = dt_helper_v.get_data();
    int* j = dt_helper_j.get_data();


    my_policy_t p{u};
    auto n_samples = u.size();

    const int N = 10;
    std::atomic<int> ecount = 0;
    bool pass = true;

    for (int i = 1; i <= N; ++i)
    {
        if (i <= 2 * n_samples && (i - 1) % n_samples != best_resource)
        {
            *j = 100;
        }
        else
        {
            *j = 0;
        }
        // we can capture all by reference
        // the inline_scheduler reports timings in submit
        // We wait but it should return immediately, since inline
        // scheduler does the work "inline".
        // The unwrapped wait type should be equal to the resource
        if constexpr (call_select_before_submit)
        {
            auto f = [&](typename oneapi::dpl::experimental::policy_traits<Policy>::resource_type q) {
                if (i <= 2 * n_samples)
                {
                    // we should be round-robining through the resources
                    if (q != u[(i - 1) % n_samples])
                    {
                        std::cout << i << ": mismatch during rr phase\n" << std::flush;
                        pass = false;
                    }
                }
                else
                {
                    if (q != u[best_resource])
                    {
                        std::cout << i << ": mismatch during prod phase " << best_resource << "\n" << std::flush;
                        pass = false;
                    }
                }
                ecount += i;
                if (*j == 0)
                {
                     return q.submit([=](sycl::handler& h){
                        h.single_task<class SingleTask1>([](){});
                     });
                }
                else
                {
                    return q.submit([=](sycl::handler& h) {
                        h.parallel_for<TestUtils::unique_kernel_name<class tune1, 0>>(
                            1000000, [=](sycl::id<1> idx) {
                                for (int j0 = 0; j0 < *j; ++j0)
                                {
                                    v[idx] += idx;
                                }
                            });
                    });
                }
            };
            auto s = oneapi::dpl::experimental::select(p, f);
            auto e = oneapi::dpl::experimental::submit(s, f);
            oneapi::dpl::experimental::wait(e);
        }
        else
        {
            // it's ok to capture by reference since we are waiting on each call
            auto s = oneapi::dpl::experimental::submit(
                p, [&](typename oneapi::dpl::experimental::policy_traits<Policy>::resource_type q) {
                    if (i <= 2 * n_samples)
                    {
                        // we should be round-robining through the resources
                        if (q != u[(i - 1) % n_samples])
                        {
                            std::cout << i << ": mismatch during rr phase\n" << std::flush;
                            pass = false;
                        }
                    }
                    else
                    {
                        if (q != u[best_resource])
                        {
                            std::cout << i << ": mismatch during prod phase " << best_resource << "\n" << std::flush;
                            pass = false;
                        }
                    }
                    ecount += i;
                    if (*j == 0)
                    {
                         return q.submit([=](sycl::handler& h){
                            h.single_task<class SingleTask2>([](){});
                         });
                    }
                    else
                    {
                        return q.submit([=](sycl::handler& h) {
                            h.parallel_for<TestUtils::unique_kernel_name<class tune2, 0>>(
                                1000000, [=](sycl::id<1> idx) {
                                    for (int j0 = 0; j0 < *j; ++j0)
                                    {
                                        v[idx] += idx;
                                    }
                                });
                        });
                    }
                });
            oneapi::dpl::experimental::wait(s);
        }

        int count = ecount.load();
        EXPECT_EQ(i * (i + 1) / 2, count, "ERROR: scheduler did not execute all tasks exactly once\n");
    }
    EXPECT_TRUE(pass, "ERROR: did not select expected resources\n");
    if constexpr (call_select_before_submit)
    {
        std::cout << "select then submit and wait on event: OK\n";
    }
    else
    {
        std::cout << "submit and wait on event: OK\n";
    }
    return 0;
}

template <bool call_select_before_submit, typename Policy, typename UniverseContainer>
int
test_auto_submit_wait_on_group(UniverseContainer u, int best_resource)
{
    using my_policy_t = Policy;

    // they are cpus so this is ok
    TestUtils::usm_data_transfer<sycl::usm::alloc::shared, double> dt_helper_v(u[0], 1000000);
    TestUtils::usm_data_transfer<sycl::usm::alloc::shared, int> dt_helper_j(u[0], 1);

    //Making v volatile so the release build does not optimize it in the for loop below
    volatile double* v = dt_helper_v.get_data();
    int* j = dt_helper_j.get_data();


    my_policy_t p{u};
    auto n_samples = u.size();

    const int N = 10;
    std::atomic<int> ecount = 0;
    bool pass = true;

    for (int i = 1; i <= N; ++i)
    {
        if (i <= 2 * n_samples && (i - 1) % n_samples != best_resource)
        {
            *j = 100;
        }
        else
        {
            *j = 0;
        }
        // we can capture all by reference
        // the inline_scheduler reports timings in submit
        // We wait but it should return immediately, since inline
        // scheduler does the work "inline".
        // The unwrapped wait type should be equal to the resource
        if constexpr (call_select_before_submit)
        {
            auto f = [&](typename oneapi::dpl::experimental::policy_traits<Policy>::resource_type q) {
                if (i <= 2 * n_samples)
                {
                    // we should be round-robining through the resources
                    if (q != u[(i - 1) % n_samples])
                    {
                        std::cout << i << ": mismatch during rr phase\n" << std::flush;
                        pass = false;
                    }
                }
                else
                {
                    if (q != u[best_resource])
                    {
                        std::cout << i << ": mismatch during prod phase " << best_resource << "\n" << std::flush;
                        pass = false;
                    }
                }
                ecount += i;
                if (*j == 0)
                {
                     return q.submit([=](sycl::handler& h){
                        h.single_task<class SingleTask3>([](){});
                     });
                }
                else
                {
                    return q.submit([=](sycl::handler& h) {
                        h.parallel_for<TestUtils::unique_kernel_name<class tune3, 0>>(
                            1000000, [=](sycl::id<1> idx) {
                                for (int j0 = 0; j0 < *j; ++j0)
                                {
                                    v[idx] += idx;
                                }
                            });
                    });
                }
            };
            auto s = oneapi::dpl::experimental::select(p, f);
            auto e = oneapi::dpl::experimental::submit(s, f);
            oneapi::dpl::experimental::wait(p.get_submission_group());
        }
        else
        {
            // it's ok to capture by reference since we are waiting on each call
            auto s = oneapi::dpl::experimental::submit(
                p, [&](typename oneapi::dpl::experimental::policy_traits<Policy>::resource_type q) {
                    if (i <= 2 * n_samples)
                    {
                        // we should be round-robining through the resources
                        if (q != u[(i - 1) % n_samples])
                        {
                            std::cout << i << ": mismatch during rr phase\n" << std::flush;
                            pass = false;
                        }
                    }
                    else
                    {
                        if (q != u[best_resource])
                        {
                            std::cout << i << ": mismatch during prod phase " << best_resource << "\n" << std::flush;
                            pass = false;
                        }
                    }
                    ecount += i;
                    if (*j == 0)
                    {
                         return q.submit([=](sycl::handler& h){
                            h.single_task<class SingleTask4>([](){});
                         });
                    }
                    else
                    {
                        return q.submit([=](sycl::handler& h) {
                            h.parallel_for<TestUtils::unique_kernel_name<class tune4, 0>>(
                                1000000, [=](sycl::id<1> idx) {
                                    for (int j0 = 0; j0 < *j; ++j0)
                                    {
                                        v[idx] += idx;
                                    }
                                });
                        });
                    }
                });
            oneapi::dpl::experimental::wait(p.get_submission_group());
        }

        int count = ecount.load();
        EXPECT_EQ(i * (i + 1) / 2, count, "ERROR: scheduler did not execute all tasks exactly once\n");
    }
    EXPECT_TRUE(pass, "ERROR: did not select expected resources\n");
    if constexpr (call_select_before_submit)
    {
        std::cout << "select then submit and wait on group: OK\n";
    }
    else
    {
        std::cout << "submit and wait on group: OK\n";
    }
    return 0;
}


template <bool call_select_before_submit, typename Policy, typename UniverseContainer>
int
test_auto_submit_and_wait(UniverseContainer u, int best_resource)
{
    using my_policy_t = Policy;

    // they are cpus so this is ok
    TestUtils::usm_data_transfer<sycl::usm::alloc::shared, double> dt_helper_v(u[0], 1000000);
    TestUtils::usm_data_transfer<sycl::usm::alloc::shared, int> dt_helper_j(u[0], 1);

    //Making v volatile so the release build does not optimize it in the for loop below
    volatile double* v = dt_helper_v.get_data();
    int* j = dt_helper_j.get_data();

    my_policy_t p{u};
    auto n_samples = u.size();

    const int N = 10;
    std::atomic<int> ecount = 0;
    bool pass = true;

    for (int i = 1; i <= N; ++i)
    {
        if (i <= 2 * n_samples && (i - 1) % n_samples != best_resource)
        {
            *j = 100;
        }
        else
        {
            *j = 0;
        }
        // we can capture all by reference
        // the inline_scheduler reports timings in submit
        // We wait but it should return immediately, since inline
        // scheduler does the work "inline".
        // The unwrapped wait type should be equal to the resource
        if constexpr (call_select_before_submit)
        {
            auto f = [&](typename oneapi::dpl::experimental::policy_traits<Policy>::resource_type q) {
                if (i <= 2 * n_samples)
                {
                    // we should be round-robining through the resources
                    if (q != u[(i - 1) % n_samples])
                    {
                        std::cout << i << ": mismatch during rr phase\n" << std::flush;
                        pass = false;
                    }
                }
                else
                {
                    if (q != u[best_resource])
                    {
                        std::cout << i << ": mismatch during prod phase " << best_resource << "\n" << std::flush;
                        pass = false;
                    }
                }
                ecount += i;
                if (*j == 0)
                {
                     return q.submit([=](sycl::handler& h){
                        h.single_task<class SingleTask5>([](){});
                     });
                }
                else
                {
                    return q.submit([=](sycl::handler& h) {
                        h.parallel_for<TestUtils::unique_kernel_name<class tune5, 0>>(
                            1000000, [=](sycl::id<1> idx) {
                                for (int j0 = 0; j0 < *j; ++j0)
                                {
                                    v[idx] += idx;
                                }
                            });
                    });
                }
            };
            auto s = oneapi::dpl::experimental::select(p, f);
            oneapi::dpl::experimental::submit_and_wait(s, f);
        }
        else
        {
            // it's ok to capture by reference since we are waiting on each call
            oneapi::dpl::experimental::submit_and_wait(
                p, [&](typename oneapi::dpl::experimental::policy_traits<Policy>::resource_type q) {
                    if (i <= 2 * n_samples)
                    {
                        // we should be round-robining through the resources
                        if (q != u[(i - 1) % n_samples])
                        {
                            std::cout << i << ": mismatch during rr phase\n" << std::flush;
                            pass = false;
                        }
                    }
                    else
                    {
                        if (q != u[best_resource])
                        {
                            std::cout << i << ": mismatch during prod phase " << best_resource << "\n" << std::flush;
                            pass = false;
                        }
                    }
                    ecount += i;
                    if (*j == 0)
                    {
                         return q.submit([=](sycl::handler& h){
                            h.single_task<class SingleTask6>([](){});
                         });
                    }
                    else
                    {
                        return q.submit([=](sycl::handler& h) {
                            h.parallel_for<TestUtils::unique_kernel_name<class tune6, 0>>(
                                1000000, [=](sycl::id<1> idx) {
                                    for (int j0 = 0; j0 < *j; ++j0)
                                    {
                                        v[idx] += idx;
                                    }
                                });
                        });
                    }
                });
        }

        int count = ecount.load();
        EXPECT_EQ(i * (i + 1) / 2, count, "ERROR: scheduler did not execute all tasks exactly once\n");
    }
    EXPECT_TRUE(pass, "ERROR: did not select expected resources\n");
    if constexpr (call_select_before_submit)
    {
        std::cout << "select then submit_and_wait: OK\n";
    }
    else
    {
        std::cout << "submit_and_wait: OK\n";
    }
    return 0;
}


template<bool use_event_profiling=false>
static inline void
build_auto_tune_universe(std::vector<sycl::queue>& u)
{
    auto prop_list = sycl::property_list{};
    if(use_event_profiling){
        prop_list = sycl::property_list{sycl::property::queue::enable_profiling()};
    }

    try
    {
        auto device_cpu1 = sycl::device(sycl::cpu_selector_v);
        sycl::queue cpu1_queue{device_cpu1, prop_list};
        u.push_back(cpu1_queue);
    }
    catch (const sycl::exception&)
    {
        std::cout << "SKIPPED: Unable to run with cpu_selector\n";
    }
    try
    {
        auto device_cpu2 = sycl::device(sycl::cpu_selector_v);
        sycl::queue cpu2_queue{device_cpu2, prop_list};
        u.push_back(cpu2_queue);
    }
    catch (const sycl::exception&)
    {
        std::cout << "SKIPPED: Unable to run with cpu_selector\n";
    }
    try
    {
        auto device_cpu3 = sycl::device(sycl::cpu_selector_v);
        sycl::queue cpu3_queue{device_cpu3, prop_list};
        u.push_back(cpu3_queue);
    }
    catch (const sycl::exception&)
    {
        std::cout << "SKIPPED: Unable to run with cpu_selector\n";
    }
    try
    {
        auto device_cpu4 = sycl::device(sycl::cpu_selector_v);
        sycl::queue cpu4_queue{device_cpu4, prop_list};
        u.push_back(cpu4_queue);
    }
    catch (const sycl::exception&)
    {
        std::cout << "SKIPPED: Unable to run with cpu_selector\n";
    }
}

#endif //TEST_DYNAMIC_SELECTION_AVAILABLE

int
main()
{
    bool bProcessed = false;

#if TEST_DYNAMIC_SELECTION_AVAILABLE
#if !ONEDPL_FPGA_DEVICE || !ONEDPL_FPGA_EMULATOR
    using policy_t = oneapi::dpl::experimental::auto_tune_policy<sycl::queue, oneapi::dpl::experimental::default_backend<sycl::queue>>;
    std::vector<sycl::queue> u1;
    std::vector<sycl::queue> u2;
    constexpr bool use_event_profiling = true;
    build_auto_tune_universe(u1);
    build_auto_tune_universe<use_event_profiling>(u2);

    if (u1.size() != 0 || u2.size() !=0 )
    {
        auto f = [u1](int i) {
            if (i <= 8)
                return u1[(i - 1) % 4];
            else
                return u1[0];
        };

        constexpr bool just_call_submit = false;
        constexpr bool call_select_before_submit = true;

        auto actual = test_auto_initialization(u1);
        actual = test_select<policy_t, decltype(u1), const decltype(f)&, true>(u1, f);
        actual = test_auto_submit_wait_on_event<just_call_submit, policy_t>(u1, 0);
        actual = test_auto_submit_wait_on_event<just_call_submit, policy_t>(u1, 1);
        actual = test_auto_submit_wait_on_event<just_call_submit, policy_t>(u1, 2);
        actual = test_auto_submit_wait_on_event<just_call_submit, policy_t>(u1, 3);
        actual = test_auto_submit_wait_on_group<just_call_submit, policy_t>(u1, 0);
        actual = test_auto_submit_wait_on_group<just_call_submit, policy_t>(u1, 1);
        actual = test_auto_submit_wait_on_group<just_call_submit, policy_t>(u1, 2);
        actual = test_auto_submit_wait_on_group<just_call_submit, policy_t>(u1, 3);
        actual = test_auto_submit_and_wait<just_call_submit, policy_t>(u1, 0);
        actual = test_auto_submit_and_wait<just_call_submit, policy_t>(u1, 1);
        actual = test_auto_submit_and_wait<just_call_submit, policy_t>(u1, 2);
        actual = test_auto_submit_and_wait<just_call_submit, policy_t>(u1, 3);
        // now select then submits
        actual = test_auto_submit_wait_on_event<call_select_before_submit, policy_t>(u1, 0);
        actual = test_auto_submit_wait_on_event<call_select_before_submit, policy_t>(u1, 1);
        actual = test_auto_submit_wait_on_event<call_select_before_submit, policy_t>(u1, 2);
        actual = test_auto_submit_wait_on_event<call_select_before_submit, policy_t>(u1, 3);
        actual = test_auto_submit_wait_on_group<call_select_before_submit, policy_t>(u1, 0);
        actual = test_auto_submit_wait_on_group<call_select_before_submit, policy_t>(u1, 1);
        actual = test_auto_submit_wait_on_group<call_select_before_submit, policy_t>(u1, 2);
        actual = test_auto_submit_wait_on_group<call_select_before_submit, policy_t>(u1, 3);
        actual = test_auto_submit_and_wait<call_select_before_submit, policy_t>(u1, 0);
        actual = test_auto_submit_and_wait<call_select_before_submit, policy_t>(u1, 1);
        actual = test_auto_submit_and_wait<call_select_before_submit, policy_t>(u1, 2);
        actual = test_auto_submit_and_wait<call_select_before_submit, policy_t>(u1, 3);
        // Use event profiling
        actual = test_auto_submit_wait_on_event<just_call_submit, policy_t>(u2, 0);
        actual = test_auto_submit_wait_on_event<just_call_submit, policy_t>(u2, 1);
        actual = test_auto_submit_wait_on_event<just_call_submit, policy_t>(u2, 2);
        actual = test_auto_submit_wait_on_event<just_call_submit, policy_t>(u2, 3);
        actual = test_auto_submit_wait_on_group<just_call_submit, policy_t>(u2, 0);
        actual = test_auto_submit_wait_on_group<just_call_submit, policy_t>(u2, 1);
        actual = test_auto_submit_wait_on_group<just_call_submit, policy_t>(u2, 2);
        actual = test_auto_submit_wait_on_group<just_call_submit, policy_t>(u2, 3);
        actual = test_auto_submit_and_wait<just_call_submit, policy_t>(u2, 0);
        actual = test_auto_submit_and_wait<just_call_submit, policy_t>(u2, 1);
        actual = test_auto_submit_and_wait<just_call_submit, policy_t>(u2, 2);
        actual = test_auto_submit_and_wait<just_call_submit, policy_t>(u2, 3);
        // now select then submits
        actual = test_auto_submit_wait_on_event<call_select_before_submit, policy_t>(u2, 0);
        actual = test_auto_submit_wait_on_event<call_select_before_submit, policy_t>(u2, 1);
        actual = test_auto_submit_wait_on_event<call_select_before_submit, policy_t>(u2, 2);
        actual = test_auto_submit_wait_on_event<call_select_before_submit, policy_t>(u2, 3);
        actual = test_auto_submit_wait_on_group<call_select_before_submit, policy_t>(u2, 0);
        actual = test_auto_submit_wait_on_group<call_select_before_submit, policy_t>(u2, 1);
        actual = test_auto_submit_wait_on_group<call_select_before_submit, policy_t>(u2, 2);
        actual = test_auto_submit_wait_on_group<call_select_before_submit, policy_t>(u2, 3);
        actual = test_auto_submit_and_wait<call_select_before_submit, policy_t>(u2, 0);
        actual = test_auto_submit_and_wait<call_select_before_submit, policy_t>(u2, 1);
        actual = test_auto_submit_and_wait<call_select_before_submit, policy_t>(u2, 2);
        actual = test_auto_submit_and_wait<call_select_before_submit, policy_t>(u2, 3);

        bProcessed = true;
    }
#endif // Devices available are CPU and GPU
#endif // TEST_DYNAMIC_SELECTION_AVAILABLE

    return TestUtils::done(bProcessed);
}
