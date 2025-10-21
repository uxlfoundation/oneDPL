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
#include "oneapi/dpl/functional"

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

// Helper to check if backend defines a wait_type
template <typename T, typename = void> //assumes wait_type does not exist
struct get_wait_type {
    using type = int; //defaults to int
};

template <typename T> //specialization if wait_type exists
struct get_wait_type<T, std::void_t<typename T::wait_type>> {
    using type = typename T::wait_type;
};

//resource providing a wait functionality
struct DummyResource 
{
    int value;

    DummyResource(int v) : value(v) {}
    bool operator==(const DummyResource& other) const 
    {
        return value == other.value;
    }

    bool operator!=(const DummyResource& other) const 
    {
        return !(*this == other);
    }

    void wait()
    {
    }
};

template <typename Policy, typename UniverseContainer, typename... Args>
int
test_dl_initialization(const UniverseContainer& u, Args&&... args)
{
    // initialize
    Policy p{u, std::forward<Args>(args)...}; //TODO:Remove need for type specification
    auto u2 = oneapi::dpl::experimental::get_resources(p);
    if (!std::equal(std::begin(u2), std::end(u2), std::begin(u)))
    {
        std::cout << "ERROR: provided resources and queried resources are not equal\n";
        return 1;
    }

    // deferred initialization
    Policy p2{oneapi::dpl::experimental::deferred_initialization};
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
    p2.initialize(u, std::forward<Args>(args)...);
    auto u3 = oneapi::dpl::experimental::get_resources(p);
    if (!std::equal(std::begin(u3), std::end(u3), std::begin(u)))
    {
        std::cout << "ERROR: reported resources and queried resources are not equal after deferred initialization\n";
        return 1;
    }

    std::cout << "initialization: OK\n" << std::flush;
    return 0;
}

template <typename CustomName, typename Policy, typename UniverseContainer, typename ResourceFunction, typename ResourceAdapter = oneapi::dpl::identity>
int
test_submit_and_wait_on_group(UniverseContainer u, ResourceFunction&& f, ResourceAdapter adapter = {})
{
    using my_policy_t = Policy;

    // This doesnt test the default initializer for policy when adapter isn't provided, but other tests do.
    my_policy_t p{u, adapter};

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
                    auto e2 = adapter(e).submit([&](sycl::handler& cgh) {
                        auto accessorA = bufferA.get_access<sycl::access::mode::read>(cgh);
                        auto accessorB = bufferB.get_access<sycl::access::mode::read>(cgh);
                        auto accessorResultMatrix = bufferResultMatrix.get_access<sycl::access::mode::write>(cgh);
                        cgh.parallel_for<TestUtils::unique_kernel_name<CustomName, 1>>(
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
                    auto e2 = adapter(e).submit([&](sycl::handler&) {
                        // for(int i=0;i<1;i++);
                    });
                    return e2;
                }
            });
        if (i > 0)
            std::this_thread::sleep_for(std::chrono::milliseconds(3));
    }
    oneapi::dpl::experimental::wait(p.get_submission_group());
    if (probability < total_items / 2)
    {
        std::cout << "ERROR: did not select expected resources\n";
        return 1;
    }
    std::cout << "submit and wait on group: OK\n";
    return 0;
}

template <typename Policy, typename UniverseContainer, typename ResourceFunction, typename... Args>
int
test_submit_and_wait_on_event(UniverseContainer u, ResourceFunction&& f, Args&&... args)
{
    using my_policy_t = Policy;
    my_policy_t p{u, std::forward<Args>(args)...};

    const int N = 6;
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
                return typename get_wait_type<typename Policy::backend_t>::type{};
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

template <typename Policy, typename UniverseContainer, typename ResourceFunction, typename... Args>
int
test_submit_and_wait(UniverseContainer u, ResourceFunction&& f, Args&&... args)
{
    using my_policy_t = Policy;
    my_policy_t p{u, std::forward<Args>(args)...};

    const int N = 6;
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
                if constexpr (std::is_same_v<
                                    typename oneapi::dpl::experimental::policy_traits<Policy>::resource_type, int>)
                    return e;
                else
                    return typename get_wait_type<typename Policy::backend_t>::type{};
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
#endif // TEST_DYNAMIC_SELECTION_AVAILABLE

#endif /* _ONEDPL_TEST_DYNAMIC_LOAD_UTILS_H */
