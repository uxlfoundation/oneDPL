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
#include <thread>
#include "oneapi/dpl/dynamic_selection"
#include "support/inline_backend.h"
#include "support/utils.h"
#include "support/barriers.h"

template <bool call_select_before_submit, typename Policy, typename UniverseContainer, typename UniverseMapping>
int
test_submit_and_wait(UniverseContainer u, UniverseMapping map, std::vector<int> actual, int best_resource)
{

    using my_policy_t = Policy;
    std::vector<int> result(u.size(), 0);
    my_policy_t p{u};
    auto n_samples = u.size();
    auto func = [&](typename oneapi::dpl::experimental::policy_traits<Policy>::resource_type e) {
        int x = map[e];
        result[x]++;
        std::this_thread::sleep_for(std::chrono::milliseconds(e));
        return typename oneapi::dpl::experimental::policy_traits<Policy>::wait_type{};
    };
    std::vector<std::thread> threads;
    int n_threads = 2;
    Barrier sync_point(n_threads);
    if(call_select_before_submit){
        auto thread_func = [&p, &func, &sync_point](){
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
    std::cout << "submit_and_wait : OK\n";
    return 0;
}
template <bool call_select_before_submit, typename Policy, typename UniverseContainer, typename UniverseMapping>
int
test_submit_and_wait_on_group(UniverseContainer u, UniverseMapping map, std::vector<int> actual, int best_resource)
{

    using my_policy_t = Policy;
    std::vector<int> result(u.size(), 0);

    my_policy_t p{u};
    auto n_samples = u.size();
    auto func = [&](typename oneapi::dpl::experimental::policy_traits<Policy>::resource_type e) {
        int x = map[e];
        result[x]++;
        std::this_thread::sleep_for(std::chrono::milliseconds(e));
        return typename oneapi::dpl::experimental::policy_traits<Policy>::wait_type{};
    };
    std::vector<std::thread> threads;
    int n_threads = 2;
    Barrier sync_point(n_threads);
    if(call_select_before_submit){
        auto thread_func = [&p, &func, &sync_point](){
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
test_submit_and_wait_on_event(UniverseContainer u, UniverseMapping map, std::vector<int> actual, int best_resource)
{

    using my_policy_t = Policy;
    std::vector<int> result(u.size(), 0);

    my_policy_t p{u};
    auto n_samples = u.size();
    auto func = [&](typename oneapi::dpl::experimental::policy_traits<Policy>::resource_type e) {
        int x = map[e];
        result[x]++;
        std::this_thread::sleep_for(std::chrono::milliseconds(e));
        return typename oneapi::dpl::experimental::policy_traits<Policy>::wait_type{};
    };
    std::vector<std::thread> threads;
    int n_threads = 2;
    Barrier sync_point(n_threads);
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
static inline void
build_universe(std::vector<int> u, std::unordered_map<int, int>& map)
{
    for(int i=0;i<u.size();i++){
        map[u[i]]=i;
    }
}

std::vector<int>
build_result(int universe_size, int total, int best_resource){
    std::vector<int> result(universe_size, 0);
    for(int i=0;i<2*universe_size;i++){
        result[i%universe_size]++;
    }
    result[best_resource]+=total-2*universe_size;
    return result;
}

template <typename Policy>
void
run_tests(std::vector<int> u, int total_elements, std::vector<int> result, int best_resource)
{
    using policy_t = Policy;

    constexpr bool just_call_submit = false;
    constexpr bool call_select_before_submit = true;
    std::unordered_map<int, int> map;
    build_universe(u, map);

    auto actual = test_submit_and_wait_on_event<call_select_before_submit, policy_t>(u, map, result, best_resource);
    actual = test_submit_and_wait_on_event<just_call_submit, policy_t>(u, map, result, best_resource);
    actual = test_submit_and_wait_on_group<call_select_before_submit, policy_t>(u, map, result, best_resource);
    actual = test_submit_and_wait_on_group<just_call_submit, policy_t>(u, map, result, best_resource);
    actual = test_submit_and_wait<call_select_before_submit, policy_t>(u, map, result, best_resource);
    actual = test_submit_and_wait<just_call_submit, policy_t>(u, map, result, best_resource);
}

int
main()
{
    using policy_t = oneapi::dpl::experimental::auto_tune_policy<TestUtils::int_inline_backend_t>;
    std::vector<int> first_resources = {1, 101, 102, 103};
    std::vector<int> second_resources = {101, 1, 102, 103};
    std::vector<int> third_resources = {101, 102, 1, 103};
    std::vector<int> fourth_resources = {101, 102, 103, 1};
    int total_elements = 16;

    run_tests<policy_t>(first_resources,  total_elements, build_result(first_resources.size(), total_elements, 0), 0);
    run_tests<policy_t>(second_resources, total_elements, build_result(second_resources.size(), total_elements, 1), 1);
    run_tests<policy_t>(third_resources, total_elements, build_result(third_resources.size(), total_elements, 2), 2);
    run_tests<policy_t>(fourth_resources, total_elements, build_result(fourth_resources.size(), total_elements, 3), 3);

    return TestUtils::done();
}
