// -*- C++ -*-
//===---------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _ONEDPL_TEST_ROUND_ROBIN_PARALLEL_UTILS_H
#define _ONEDPL_TEST_ROUND_ROBIN_PARALLEL_UTILS_H

#include "support/test_config.h"
#include <unordered_map>
#include <thread>
#include "oneapi/dpl/dynamic_selection"
#include "support/utils.h"
#include "support/barriers.h"

template <bool call_select_before_submit, typename Policy, typename UniverseContainer, typename UniverseMapping>
int
test_submit_and_wait(UniverseContainer u, UniverseMapping map, std::vector<int> actual)
{
    using my_policy_t = Policy;
    my_policy_t p(u);
    std::vector<int> result(u.size(), 0);

    std::vector<std::thread> threads;
    auto func = [&result,&map](typename oneapi::dpl::experimental::policy_traits<Policy>::resource_type e) {
        int x = map[e];
        result[x]++;
        if constexpr (std::is_same_v<typename oneapi::dpl::experimental::policy_traits<Policy>::resource_type,
                                     int>)
            return e;
        else
            return typename oneapi::dpl::experimental::policy_traits<Policy>::wait_type{};
    };
    int n_threads = 5;
    Barrier sync_point(n_threads);
    if(call_select_before_submit){
        auto thread_func = [&p, &func, &sync_point](){
            for(int i=0;i<10;i++){
                auto s = oneapi::dpl::experimental::select(p);
                oneapi::dpl::experimental::submit_and_wait(s, func);

            }
            sync_point.arrive_and_wait();
        };

        for(int i=0;i<n_threads;i++){
            threads.emplace_back(thread_func);
        }
    }
    else{
        auto thread_func = [&p, &func, &sync_point](){
            for(int i=0;i<10;i++){
                oneapi::dpl::experimental::submit_and_wait(p, func);
            }
            sync_point.arrive_and_wait();
        };

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
test_submit_and_wait_on_group(UniverseContainer u, UniverseMapping map, std::vector<int> actual)
{
    using my_policy_t = Policy;
    my_policy_t p(u);
    std::vector<int> result(u.size(), 0);

    std::vector<std::thread> threads;
    auto func = [&result,&map](typename oneapi::dpl::experimental::policy_traits<Policy>::resource_type e) {
        int x = map[e];
        result[x]++;
        if constexpr (std::is_same_v<typename oneapi::dpl::experimental::policy_traits<Policy>::resource_type,
                                     int>)
            return e;
        else
            return typename oneapi::dpl::experimental::policy_traits<Policy>::wait_type{};
    };
    int n_threads = 5;
    Barrier sync_point(n_threads);
    if(call_select_before_submit){
        auto thread_func = [&p, &func, &sync_point](){
            for(int i=0;i<10;i++){
                auto s = oneapi::dpl::experimental::select(p);

                auto w = oneapi::dpl::experimental::submit(s, func);
            }
            sync_point.arrive_and_wait();
        };

        for(int i=0;i<n_threads;i++){
            threads.emplace_back(thread_func);
        }
    }
    else{
        auto thread_func = [&p, &func, &sync_point](){
            for(int i=0;i<10;i++){
                auto w = oneapi::dpl::experimental::submit(p, func);
            }
            sync_point.arrive_and_wait();
        };

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
test_submit_and_wait_on_event(UniverseContainer u, UniverseMapping map, std::vector<int> actual)
{
    using my_policy_t = Policy;
    my_policy_t p(u);
    std::vector<int> result(u.size(), 0);

    std::vector<std::thread> threads;
    auto func = [&result,&map](typename oneapi::dpl::experimental::policy_traits<Policy>::resource_type e) {
        int x = map[e];
        result[x]++;

        if constexpr (std::is_same_v<typename oneapi::dpl::experimental::policy_traits<Policy>::resource_type,
                                     int>)
            return e;
        else
            return typename oneapi::dpl::experimental::policy_traits<Policy>::wait_type{};
    };
    int n_threads = 5;
    Barrier sync_point(n_threads);
    if(call_select_before_submit){
        auto thread_func = [&p, &func, &sync_point](){
            for(int i=0;i<10;i++){
                auto s = oneapi::dpl::experimental::select(p);

                auto w = oneapi::dpl::experimental::submit(s, func);
                oneapi::dpl::experimental::wait(w);
            }
            sync_point.arrive_and_wait();
        };

        for(int i=0;i<n_threads;i++){
            threads.emplace_back(thread_func);
        }
    }
    else{
        auto thread_func = [&p, &func, &sync_point](){
            for(int i=0;i<10;i++){
                auto w = oneapi::dpl::experimental::submit(p, func);
                oneapi::dpl::experimental::wait(w);
            }
            sync_point.arrive_and_wait();
        };

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

static inline auto
build_result(int universe_size, int count){
    std::vector<int> result(universe_size, 0);
    for(int i=0;i<count;i++){
        result[i%universe_size]++;
    }
    return result;
}
#endif // _ONEDPL_TEST_FIXED_RESOURCE_PARALLEL_UTILS_H
