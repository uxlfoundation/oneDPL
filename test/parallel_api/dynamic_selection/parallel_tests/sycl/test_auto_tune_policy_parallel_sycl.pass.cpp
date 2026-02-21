// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "support/test_config.h"

#include <iostream>
#include <unordered_map>
#include <thread>
#include "oneapi/dpl/dynamic_selection"
#include "support/utils.h"
#include "support/barriers.h"

template <bool call_select_before_submit, typename Policy, typename UniverseContainer, typename UniverseMapping>
int
test_submit_and_wait_on_group(UniverseContainer u, UniverseMapping map, std::vector<int> actual, int best_resource)
{

    using my_policy_t = Policy;
    std::vector<int> result(u.size(), 0);

    // they are cpus so this is ok
    double* v = sycl::malloc_shared<double>(1000000, u[0]);

    my_policy_t p{u};
    auto n_samples = u.size();
    std::atomic<int> counter = 0;
    auto func = [&](typename oneapi::dpl::experimental::policy_traits<Policy>::resource_type e) {
        int x = map[e];
        result[x]++;
        int index = counter++;
        auto target = index%u.size();
        if (index<2*u.size() && target!=best_resource)
        {
            return e.submit([=](sycl::handler& h) {
                h.parallel_for(
                    1000000, [=](sycl::id<1> idx) {
                        for (int j0 = 0; j0 < 10000; ++j0)
                        {
                            v[idx] += idx;
                        }
                    });
            });
        }
        else
        {
            return e.submit([=](sycl::handler& h){
                            h.single_task([](){});
                         });
        }
    };
    std::vector<std::thread> threads;
    int n_threads = 2;
    Barrier sync_point(n_threads);
    if(call_select_before_submit){
        auto thread_func = [&p, &func,&sync_point](){
            for(int i=0;i<4;i++){
                auto s = oneapi::dpl::experimental::select(p, func);

                auto w = oneapi::dpl::experimental::submit(s, func);
            }
            sync_point.arrive_and_wait();
        };
        for(int i=0;i<n_threads;i++){
            threads.emplace_back(thread_func);
        }
        oneapi::dpl::experimental::wait(p.get_submission_group());
        for(auto& thread : threads){
            thread.join();
        }
        threads.clear();
        for(int i=0;i<n_threads;i++){
            threads.emplace_back(thread_func);
        }
    }
    else{

        auto thread_func = [&p, &func, &sync_point](){
            for(int i=0;i<4;i++){
                auto w = oneapi::dpl::experimental::submit(p, func);
            }
            sync_point.arrive_and_wait();
        };

        for(int i=0;i<n_threads;i++){
            threads.emplace_back(thread_func);
        }
        oneapi::dpl::experimental::wait(p.get_submission_group());
        for(auto& thread : threads){
            thread.join();
        }
        threads.clear();
        for(int i=0;i<n_threads;i++){
            threads.emplace_back(thread_func);
        }
    }

    oneapi::dpl::experimental::wait(p.get_submission_group());
    for(auto& thread : threads){
        thread.join();
    }
    bool pass = (actual == result);
    if (!pass)
    {
        std::cout << "ERROR: did not select expected resources\n";
        return 1;
    }
    std::cout << "submit_and_wait_on_group: OK\n";
    return 0;
}
template <bool call_select_before_submit, typename Policy, typename UniverseContainer, typename UniverseMapping>
int
test_submit_and_wait(UniverseContainer u, UniverseMapping map, std::vector<int> actual,int best_resource)
{

    using my_policy_t = Policy;
    std::vector<int> result(u.size(), 0);

    // they are cpus so this is ok
    double* v = sycl::malloc_shared<double>(1000000, u[0]);

    my_policy_t p{u};
    auto n_samples = u.size();

    std::atomic<int> counter = 0;
    auto func = [&](typename oneapi::dpl::experimental::policy_traits<Policy>::resource_type e) {
        int x = map[e];
        result[x]++;
        int index = counter++;
        auto target = index%u.size();
        if (index<2*u.size() && target!=best_resource)
        {
            return e.submit([=](sycl::handler& h) {
                h.parallel_for(
                    1000000, [=](sycl::id<1> idx) {
                        for (int j0 = 0; j0 < 10000; ++j0)
                        {
                            v[idx] += idx;
                        }
                    });
            });
        }
        else
        {
            return e.submit([=](sycl::handler& h){
                            h.single_task([](){});
                         });
        }
    };

    std::vector<std::thread> threads;
    int n_threads = 2;
    Barrier sync_point(n_threads);
    if(call_select_before_submit){
        auto thread_func = [&p, &func,&sync_point](){
            for(int i=0;i<4;i++){
                auto s = oneapi::dpl::experimental::select(p, func);
                oneapi::dpl::experimental::submit_and_wait(s, func);
            }
            sync_point.arrive_and_wait();
        };

        for(int i=0;i<n_threads;i++){
            threads.emplace_back(thread_func);
        }
        for(auto& thread : threads){
            thread.join();
        }
        threads.clear();
        for(int i=0;i<n_threads;i++){
            threads.emplace_back(thread_func);
        }

    }
    else{

        auto thread_func = [&p, &func, &sync_point](){
            for(int i=0;i<4;i++){
                oneapi::dpl::experimental::submit_and_wait(p, func);
            }
            sync_point.arrive_and_wait();
        };

        for(int i=0;i<n_threads;i++){
            threads.emplace_back(thread_func);
        }
        for(auto& thread : threads){
            thread.join();
        }
        threads.clear();
        for(int i=0;i<n_threads;i++){
            threads.emplace_back(thread_func);
        }
    }

    for(auto& thread : threads){
        thread.join();
    }


    bool pass = (actual == result);
    if (!pass)
    {
        std::cout << "ERROR: did not select expected resources\n";
        return 1;
    }
    std::cout << "submit_and_wait: OK\n";
    return 0;
}
template <bool call_select_before_submit, typename Policy, typename UniverseContainer, typename UniverseMapping>
int
test_submit_and_wait_on_event(UniverseContainer u, UniverseMapping map, std::vector<int> actual, int best_resource)
{

    using my_policy_t = Policy;
    std::vector<int> result(u.size(), 0);
    int n_threads = 2;
    Barrier sync_point(n_threads);
    // they are cpus so this is ok
    double* v = sycl::malloc_shared<double>(1000000, u[0]);

    my_policy_t p{u};
    auto n_samples = u.size();
    std::atomic<int> counter = 0;
    auto func = [&](typename oneapi::dpl::experimental::policy_traits<Policy>::resource_type e) {
        int x = map[e];
        result[x]++;
        int index = counter++;
        auto target = index%u.size();
        if (index<2*u.size() && target!=best_resource)
        {
            return e.submit([=](sycl::handler& h) {
                h.parallel_for(
                    1000000, [=](sycl::id<1> idx) {
                        for (int j0 = 0; j0 < 10000; ++j0)
                        {
                            v[idx] += idx;
                        }
                    });
            });
        }
        else
        {
            return e.submit([=](sycl::handler& h){
                            h.single_task([](){});
                         });
        }
    };

    std::vector<std::thread> threads;
    if(call_select_before_submit){
        auto thread_func = [&p, &func, &sync_point](){
            for(int i=0;i<4;i++){
                auto s = oneapi::dpl::experimental::select(p, func);

                auto w = oneapi::dpl::experimental::submit(s, func);
                oneapi::dpl::experimental::wait(w);
            }
            sync_point.arrive_and_wait();
        };

        for(int i=0;i<n_threads;i++){
            threads.emplace_back(thread_func);
        }


        for(auto& thread : threads){
            thread.join();
        }
        threads.clear();

        for(int i=0;i<n_threads;i++){
            threads.emplace_back(thread_func);
        }

    }
    else{

        auto thread_func = [&p, &func, &sync_point](){
            for(int i=0;i<4;i++){
                auto w = oneapi::dpl::experimental::submit(p, func);
                oneapi::dpl::experimental::wait(w);
            }
            sync_point.arrive_and_wait();
        };

        for(int i=0;i<n_threads;i++){
            threads.emplace_back(thread_func);
        }
        for(auto& thread : threads){
            thread.join();
        }
        threads.clear();

        for(int i=0;i<n_threads;i++){
            threads.emplace_back(thread_func);
        }
    }

    for(auto& thread : threads){
        thread.join();
    }
    bool pass = (actual == result);
    if (!pass)
    {
        std::cout << "ERROR: did not select expected resources\n";
        return 1;
    }
    std::cout << "submit_and_wait_on_event: OK\n";
    return 0;
}
#if TEST_DYNAMIC_SELECTION_AVAILABLE

template<bool use_event_profiling=false>
static inline void
build_universe(std::vector<sycl::queue>& u, std::unordered_map<sycl::queue, int>& map)
{
    int i=0;
    auto prop_list = sycl::property_list{};
    if(use_event_profiling){
        prop_list = sycl::property_list{sycl::property::queue::enable_profiling()};
    }
    try
    {
        auto device_cpu1 = sycl::device(sycl::cpu_selector_v);
        sycl::queue cpu1_queue{device_cpu1, prop_list};
        u.push_back(cpu1_queue);
        map[cpu1_queue]=i++;
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
        map[cpu2_queue]=i++;
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
        map[cpu3_queue]=i++;
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
        map[cpu4_queue]=i++;
    }
    catch (const sycl::exception&)
    {
        std::cout << "SKIPPED: Unable to run with cpu_selector\n";
    }
}
#endif

std::vector<int> build_result(int universe_size, int total, int best_resource){
    std::vector<int> result(universe_size, 0);
    for(int i=0;i<2*universe_size;i++){
        result[i%universe_size]++;
    }
    result[best_resource]+=total-2*universe_size;
    return result;
}
int
main()
{
    bool bProcessed = false;

#if TEST_DYNAMIC_SELECTION_AVAILABLE
#if !ONEDPL_FPGA_DEVICE || !ONEDPL_FPGA_EMULATOR
    using policy_t = oneapi::dpl::experimental::auto_tune_policy<oneapi::dpl::experimental::sycl_backend>;
    std::vector<sycl::queue> u1, u2;
    std::unordered_map<sycl::queue, int> map;
    constexpr bool use_event_profiling = true;
    build_universe(u1, map);
    build_universe<use_event_profiling>(u2, map);
    auto n = u1.size();
    int total_elements = 16;

    //If building the universe is not a success, return
    if (n != 0)
    {

        constexpr bool just_call_submit = false;
        constexpr bool call_select_before_submit = true;

        auto actual = test_submit_and_wait_on_event<call_select_before_submit, policy_t>(u1, map, build_result(n,total_elements, 0), 0);
        actual = test_submit_and_wait_on_event<call_select_before_submit, policy_t>(u1, map, build_result(n,total_elements, 1), 1);
        actual = test_submit_and_wait_on_event<call_select_before_submit, policy_t>(u1, map, build_result(n,total_elements, 2), 2);
        actual = test_submit_and_wait_on_event<call_select_before_submit, policy_t>(u1, map, build_result(n,total_elements, 3), 3);
        actual = test_submit_and_wait_on_event<just_call_submit, policy_t>(u1, map, build_result(n,total_elements,0), 0);
        actual = test_submit_and_wait_on_event<just_call_submit, policy_t>(u1, map, build_result(n,total_elements,1), 1);
        actual = test_submit_and_wait_on_event<just_call_submit, policy_t>(u1, map, build_result(n,total_elements,2), 2);
        actual = test_submit_and_wait_on_event<just_call_submit, policy_t>(u1, map, build_result(n,total_elements,3), 3);
        actual = test_submit_and_wait_on_group<just_call_submit, policy_t>(u1, map, build_result(n, total_elements, 0), 0);
        actual = test_submit_and_wait_on_group<just_call_submit, policy_t>(u1, map, build_result(n, total_elements, 1), 1);
        actual = test_submit_and_wait_on_group<just_call_submit, policy_t>(u1, map, build_result(n, total_elements, 2), 2);
        actual = test_submit_and_wait_on_group<just_call_submit, policy_t>(u1, map, build_result(n, total_elements, 3), 3);
        actual = test_submit_and_wait_on_group<call_select_before_submit, policy_t>(u1, map, build_result(n, total_elements, 0), 0);
        actual = test_submit_and_wait_on_group<call_select_before_submit, policy_t>(u1, map, build_result(n, total_elements, 1), 1);
        actual = test_submit_and_wait_on_group<call_select_before_submit, policy_t>(u1, map, build_result(n, total_elements, 2), 2);
        actual = test_submit_and_wait_on_group<call_select_before_submit, policy_t>(u1, map, build_result(n, total_elements, 3), 3);
        actual = test_submit_and_wait<just_call_submit, policy_t>(u1, map, build_result(n, total_elements, 0), 0);
        actual = test_submit_and_wait<just_call_submit, policy_t>(u1, map, build_result(n, total_elements, 1), 1);
        actual = test_submit_and_wait<just_call_submit, policy_t>(u1, map, build_result(n, total_elements, 2), 2);
        actual = test_submit_and_wait<just_call_submit, policy_t>(u1, map, build_result(n, total_elements, 3), 3);
        actual = test_submit_and_wait<call_select_before_submit, policy_t>(u1, map, build_result(n, total_elements, 0), 0);
        actual = test_submit_and_wait<call_select_before_submit, policy_t>(u1, map, build_result(n, total_elements, 1), 1);
        actual = test_submit_and_wait<call_select_before_submit, policy_t>(u1, map, build_result(n, total_elements, 2), 2);
        actual = test_submit_and_wait<call_select_before_submit, policy_t>(u1, map, build_result(n, total_elements, 3), 3);
          //Use Event Profiling
        actual = test_submit_and_wait_on_event<call_select_before_submit, policy_t>(u2, map, build_result(n,total_elements, 0), 0);
        actual = test_submit_and_wait_on_event<call_select_before_submit, policy_t>(u2, map, build_result(n,total_elements, 1), 1);
        actual = test_submit_and_wait_on_event<call_select_before_submit, policy_t>(u2, map, build_result(n,total_elements, 2), 2);
        actual = test_submit_and_wait_on_event<call_select_before_submit, policy_t>(u2, map, build_result(n,total_elements, 3), 3);
        actual = test_submit_and_wait_on_event<just_call_submit, policy_t>(u2, map, build_result(n,total_elements,0), 0);
        actual = test_submit_and_wait_on_event<just_call_submit, policy_t>(u2, map, build_result(n,total_elements,1), 1);
        actual = test_submit_and_wait_on_event<just_call_submit, policy_t>(u2, map, build_result(n,total_elements,2), 2);
        actual = test_submit_and_wait_on_event<just_call_submit, policy_t>(u2, map, build_result(n,total_elements,3), 3);
        actual = test_submit_and_wait_on_group<just_call_submit, policy_t>(u1, map, build_result(n, total_elements, 0), 0);
        actual = test_submit_and_wait_on_group<just_call_submit, policy_t>(u1, map, build_result(n, total_elements, 1), 1);
        actual = test_submit_and_wait_on_group<just_call_submit, policy_t>(u1, map, build_result(n, total_elements, 2), 2);
        actual = test_submit_and_wait_on_group<just_call_submit, policy_t>(u1, map, build_result(n, total_elements, 3), 3);
        actual = test_submit_and_wait_on_group<call_select_before_submit, policy_t>(u1, map, build_result(n, total_elements, 0), 0);
        actual = test_submit_and_wait_on_group<call_select_before_submit, policy_t>(u1, map, build_result(n, total_elements, 1), 1);
        actual = test_submit_and_wait_on_group<call_select_before_submit, policy_t>(u1, map, build_result(n, total_elements, 2), 2);
        actual = test_submit_and_wait_on_group<call_select_before_submit, policy_t>(u1, map, build_result(n, total_elements, 3), 3);
        actual = test_submit_and_wait<just_call_submit, policy_t>(u2, map, build_result(n, total_elements, 0), 0);
        actual = test_submit_and_wait<just_call_submit, policy_t>(u2, map, build_result(n, total_elements, 1), 1);
        actual = test_submit_and_wait<just_call_submit, policy_t>(u2, map, build_result(n, total_elements, 2), 2);
        actual = test_submit_and_wait<just_call_submit, policy_t>(u2, map, build_result(n, total_elements, 3), 3);
        actual = test_submit_and_wait<call_select_before_submit, policy_t>(u2, map, build_result(n, total_elements, 0), 0);
        actual = test_submit_and_wait<call_select_before_submit, policy_t>(u2, map, build_result(n, total_elements, 1), 1);
        actual = test_submit_and_wait<call_select_before_submit, policy_t>(u2, map, build_result(n, total_elements, 2), 2);
        actual = test_submit_and_wait<call_select_before_submit, policy_t>(u2, map, build_result(n, total_elements, 3), 3);

        bProcessed = true;
    }
#endif // Devices available are CPU and GPU
#endif // TEST_DYNAMIC_SELECTION_AVAILABLE

    return TestUtils::done(bProcessed);
}
