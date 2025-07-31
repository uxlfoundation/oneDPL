// -*- C++ -*-
//===---------------------------------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _ONEDPL_TEST_DYNAMIC_LOAD_UTILS_H
#define _ONEDPL_TEST_DYNAMIC_LOAD_UTILS_H

#include "support/test_config.h"

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

int
test_dl_initialization(const std::vector<sycl::queue>& u)
{
    // initialize
    oneapi::dpl::experimental::dynamic_load_policy<sycl::queue, oneapi::dpl::experimental::empty_extra_resource> p{u}; //TODO:Remove need for type specification
    auto u2 = oneapi::dpl::experimental::get_resources(p);
    if (!std::equal(std::begin(u2), std::end(u2), std::begin(u)))
    {
        std::cout << "ERROR: provided resources and queried resources are not equal\n";
        return 1;
    }

    // deferred initialization
    oneapi::dpl::experimental::dynamic_load_policy<sycl::queue> p2{oneapi::dpl::experimental::deferred_initialization};
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
test_submit_and_wait_on_group(UniverseContainer u, ResourceFunction&& f)
{
    using my_policy_t = Policy;
    my_policy_t p{u};

    // Do a matrix multiply operation with each work item processing a row of the result matrix

    constexpr size_t rows_a = 1000;
    constexpr size_t cols_a = 100;
    constexpr size_t rows_b = cols_a;
    constexpr size_t cols_b = 200;
    constexpr size_t rows_c = rows_a;
    constexpr size_t cols_c = cols_b;
    
    std::vector<int> a(rows_a * cols_a);
    std::vector<int> b(rows_b * cols_b);
    std::vector<int> resultMatrix(rows_a * cols_b);

    std::default_random_engine generator;
    std::uniform_int_distribution<int> distribution(1, 10);

    // fill each matrix with random data
    for (size_t a_idx = 0; a_idx < rows_a * cols_a; ++a_idx)
    {
        a[a_idx] = distribution(generator);
    }

    for (size_t b_idx = 0; b_idx < rows_b * cols_b; ++b_idx)
    {
        b[b_idx] = distribution(generator);
    }

    sycl::buffer<int, 2> bufferA(a.data(), sycl::range<2>(rows_a, cols_a));
    sycl::buffer<int, 2> bufferB(b.data(), sycl::range<2>(rows_b, cols_b));
    sycl::buffer<int, 2> bufferResultMatrix(resultMatrix.data(), sycl::range<2>(rows_c, cols_c));

    std::atomic<int> probability = 0;
    size_t total_items = 6;
    if constexpr (call_select_before_submit)
    {
        for (int i = 0; i < total_items; i++)
        {
            int target = i % u.size();
            auto test_resource = f(i);
            auto func = [&](typename Policy::resource_type e) {
                if (e == test_resource)
                {
                    probability.fetch_add(1);
                }
                if (target == 0)
                {
                    auto e2 = e.submit([&](sycl::handler& cgh) {
                        auto accessorA = bufferA.get_access<sycl::access::mode::read>(cgh);
                        auto accessorB = bufferB.get_access<sycl::access::mode::read>(cgh);
                        auto accessorResultMatrix = bufferResultMatrix.get_access<sycl::access::mode::write>(cgh);
                        cgh.parallel_for<TestUtils::unique_kernel_name<class load2, 0>>(
                            sycl::range<1>(rows_c), [=](sycl::item<1> row_c) {
                                for (size_t col_c = 0; col_c < cols_c; ++col_c)
                                {
                                    int dotProduct = 0;
                                    for (size_t inner_idx = 0; inner_idx < cols_a; ++inner_idx)
                                    {
                                        dotProduct += accessorA[row_c][inner_idx] * accessorB[inner_idx][col_c];
                                    }
                                    accessorResultMatrix[row_c][col_c] = dotProduct;
                                }
                            });
                    });
                    return e2;
                }
                else
                {
                    auto e2 = e.submit([&](sycl::handler&) {});
                    return e2;
                }
            };
            auto s = oneapi::dpl::experimental::select(p, func);
            auto e = oneapi::dpl::experimental::submit(s, func);
            if (i > 0)
                std::this_thread::sleep_for(std::chrono::milliseconds(3));
        }
        oneapi::dpl::experimental::wait(p.get_submission_group());
    }
    else
    {
        for (int i = 0; i < total_items; ++i)
        {
            int target = i % u.size();
            auto test_resource = f(i);
            oneapi::dpl::experimental::submit(
                p, [&](typename oneapi::dpl::experimental::policy_traits<Policy>::resource_type e) {
                    if (e == test_resource)
                    {
                        probability.fetch_add(1);
                    }
                    if (target == 0)
                    {
                        auto e2 = e.submit([&](sycl::handler& cgh) {
                            auto accessorA = bufferA.get_access<sycl::access::mode::read>(cgh);
                            auto accessorB = bufferB.get_access<sycl::access::mode::read>(cgh);
                            auto accessorResultMatrix = bufferResultMatrix.get_access<sycl::access::mode::write>(cgh);
                            cgh.parallel_for<TestUtils::unique_kernel_name<class load1, 0>>(
                                sycl::range<1>(rows_c), [=](sycl::item<1> row_c) {
                                    for (size_t col_c = 0; col_c < cols_c; ++col_c)
                                    {
                                        int dotProduct = 0;
                                        for (size_t inner_idx = 0; inner_idx < cols_a; ++inner_idx)
                                        {
                                            dotProduct += accessorA[row_c][inner_idx] * accessorB[inner_idx][col_c];
                                        }
                                        accessorResultMatrix[row_c][col_c] = dotProduct;
                                    }
                                });
                        });
                        return e2;
                    }
                    else
                    {
                        auto e2 = e.submit([&](sycl::handler&) {
                            // for(int i=0;i<1;i++);
                        });
                        return e2;
                    }
                });
            if (i > 0)
                std::this_thread::sleep_for(std::chrono::milliseconds(3));
        }
        oneapi::dpl::experimental::wait(p.get_submission_group());
    }
    if (probability < total_items / 2)
    {
        std::cout << "ERROR: did not select expected resources\n";
        return 1;
    }
    std::cout << "submit and wait on group: OK\n";
    return 0;
}

template <bool call_select_before_submit, typename Policy, typename UniverseContainer, typename ResourceFunction>
int
test_submit_and_wait_on_event(UniverseContainer u, ResourceFunction&& f)
{
    using my_policy_t = Policy;
    my_policy_t p{u};

    const int N = 6;
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
test_submit_and_wait(UniverseContainer u, ResourceFunction&& f)
{
    using my_policy_t = Policy;
    my_policy_t p{u};

    const int N = 6;
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


template <bool call_select_before_submit, typename Policy, typename UniverseContainer, typename ExtraUniverseContainer, typename ResourceFunction, typename ExtraResourceFunction>
int
test_extra_resource_submit_and_wait(UniverseContainer u, ExtraUniverseContainer v, ResourceFunction&& f, ExtraResourceFunction&& ef)
{
    //std::cout<<"testing extra resource..., vsize:"<<v.size()<<"\n";
    using my_policy_t = Policy;
    my_policy_t p{u, v};
    //std::cout<<"initialized\n";
    const int N = 6;
    std::atomic<int> ecount = 0;
    bool pass = true;

    if constexpr (call_select_before_submit)
    {
//        std::cout<<"call before submit\n";
        for (int i = 1; i <= N; ++i)
        {
            auto test_resource = f(i);
            auto test_extra_resource = ef(i);
            auto func = [&pass, test_resource, test_extra_resource, &ecount,
                         i](typename oneapi::dpl::experimental::policy_traits<Policy>::resource_type e, typename oneapi::dpl::experimental::policy_traits<Policy>::extra_resource_type ex) {
                if (e != test_resource || ex != test_extra_resource)
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
//        std::cout<<" submit and wait\n";
        for (int i = 1; i <= N; ++i)
        {
            auto test_resource = f(i);
            auto test_extra_resource = ef(i);
            oneapi::dpl::experimental::submit_and_wait(
                p, [&pass, &ecount, test_resource,
                    test_extra_resource, i](typename oneapi::dpl::experimental::policy_traits<Policy>::resource_type e, typename oneapi::dpl::experimental::policy_traits<Policy>::extra_resource_type ex) {
                    if (e != test_resource || ex != test_extra_resource)
                    {
                        std::cout<<"ERROR: did not select expected resources\n";
                        //std::cout<<" ex: "<<ex<<" test_extra_resource: "<<test_extra_resource<<"\n";
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
#endif // TEST_DYNAMIC_SELECTION_AVAILABLE

#endif /* _ONEDPL_TEST_DYNAMIC_LOAD_UTILS_H */
