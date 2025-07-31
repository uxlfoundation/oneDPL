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

#include "support/inline_backend.h"
#include "support/utils.h"
#include "support/barriers.h"

template <bool call_select_before_submit, typename Policy, typename UniverseContainer, typename UniverseMapping>
int
test_submit_and_wait(UniverseContainer u, UniverseMapping map, int best_resource)
{
    using my_policy_t = Policy;
    my_policy_t p(u);
    std::vector<int> result(u.size(), 0);

    std::vector<std::thread> threads;
    auto func = [&result,&map](typename oneapi::dpl::experimental::policy_traits<Policy>::resource_type e) {
        int x = map[e];
        result[x]++;
        return e;
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
    auto result_element = std::distance(result.begin(),std::max_element(result.begin(), result.end()));
    EXPECT_TRUE(result_element==best_resource, "ERROR : did not select expected resources\n");
    std::cout<<"Submit and wait on event : OK\n";
    return 0;
}
template <bool call_select_before_submit, typename Policy, typename UniverseContainer, typename UniverseMapping>
int
test_submit_and_wait_on_group(UniverseContainer u, UniverseMapping map, int best_resource)
{
    using my_policy_t = Policy;
    my_policy_t p(u);
    std::vector<int> result(u.size(), 0);

    std::vector<std::thread> threads;
    auto func = [&result,&map](typename oneapi::dpl::experimental::policy_traits<Policy>::resource_type e) {
        int x = map[e];
        result[x]++;
        return e;
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

        for(int i=0;i<5;i++){
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

    auto result_element = std::distance(result.begin(),std::max_element(result.begin(), result.end()));
    EXPECT_TRUE(result_element==best_resource, "ERROR : did not select expected resources\n");
    std::cout<<"Submit and wait on event : OK\n";
    return 0;
}
template <bool call_select_before_submit, typename Policy, typename UniverseContainer, typename UniverseMap>
int
test_submit_and_wait_on_event(UniverseContainer u, UniverseMap&& map, int best_resource)
{
    using my_policy_t = Policy;
    my_policy_t p(u);
    std::vector<int> result(u.size(), 0);

    auto func = [&result,&map](typename oneapi::dpl::experimental::policy_traits<Policy>::resource_type e) {
        int x = map[e];
        result[x]++;
        return e;
    };

    std::vector<std::thread> threads;
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

    auto result_element = std::distance(result.begin(),std::max_element(result.begin(), result.end()));
    EXPECT_TRUE(result_element==best_resource, "ERROR : did not select expected resources\n");
    std::cout<<"Submit and wait on event : OK\n";
    return 0;
}
#if TEST_DYNAMIC_SELECTION_AVAILABLE
static inline void
build_universe(std::vector<int>& u, std::unordered_map<int, int>& map)
{
    for(int i=0;i<u.size();i++){
        map[u[i]]=i;
    }
}
#endif // TEST_DYNAMIC_SELECTION_AVAILABLE
int
main()
{
    using policy_t = oneapi::dpl::experimental::dynamic_load_policy<TestUtils::int_inline_backend_t>;
    std::unordered_map<int, int> map;
    std::vector<int> u{4, 5, 6, 7};
    build_universe(u, map);
    int best_resource=0;
    constexpr bool just_call_submit = false;
    constexpr bool call_select_before_submit = true;

    auto actual = test_submit_and_wait_on_event<just_call_submit, policy_t>(u, map, best_resource);
    actual = test_submit_and_wait_on_event<call_select_before_submit, policy_t>(u, map, best_resource);
    actual = test_submit_and_wait<just_call_submit, policy_t>(u, map, best_resource);
    actual = test_submit_and_wait<call_select_before_submit, policy_t>(u, map, best_resource);
    actual = test_submit_and_wait_on_group<just_call_submit, policy_t>(u, map, best_resource);
    actual = test_submit_and_wait_on_group<call_select_before_submit, policy_t>(u, map, best_resource);

    return TestUtils::done();
}
