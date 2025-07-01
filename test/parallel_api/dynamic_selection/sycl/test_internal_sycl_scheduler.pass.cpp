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
#include <atomic>
#include <iostream>
#include "support/utils.h"

#if TEST_DYNAMIC_SELECTION_AVAILABLE
class fake_selection_handle_t
{
    sycl::queue q_;

  public:
    fake_selection_handle_t(sycl::queue q = sycl::queue(sycl::default_selector_v)) : q_(q) {}
    auto
    unwrap()
    {
        return q_;
    }
};

int
test_cout()
{
    //oneapi::dpl::experimental::sycl_backend s;
    //oneapi::dpl::experimental::sycl_backend::execution_resource_t e;
    oneapi::dpl::experimental::default_backend<sycl::queue> s;
    oneapi::dpl::experimental::default_backend<sycl::queue>::execution_resource_t e;
    return 0;
}

int
test_submit_and_wait_on_scheduler()
{
    const int N = 100;
    oneapi::dpl::experimental::sycl_backend s;
    fake_selection_handle_t h;

    std::atomic<int> ecount = 0;

    for (int i = 1; i <= N; ++i)
    {
        s.submit(
            h,
            [&](sycl::queue, int i) {
                ecount += i;
                return sycl::event{};
            },
            i);
    }
    s.get_submission_group().wait();
    int count = ecount.load();
    EXPECT_EQ(N * (N + 1) / 2, count, "ERROR: scheduler did not execute all tasks exactly once\n");
    std::cout << "wait_on_scheduler: OK\n";
    return 0;
}

int
test_submit_and_wait_on_scheduler_single_element()
{
    const int N = 1;
    oneapi::dpl::experimental::sycl_backend s;
    fake_selection_handle_t h;

    std::atomic<int> ecount = 0;

    for (int i = 1; i <= N; ++i)
    {
        s.submit(
            h,
            [&](sycl::queue, int i) {
                ecount += i;
                return sycl::event{};
            },
            i);
    }
    s.get_submission_group().wait();
    int count = ecount.load();
    EXPECT_EQ(1, count, "ERROR: scheduler did not execute all tasks exactly once\n");
    std::cout << "wait_on_scheduler single element: OK\n";
    return 0;
}

int
test_submit_and_wait_on_scheduler_empty()
{
    const int N = 0;
    oneapi::dpl::experimental::sycl_backend s;
    fake_selection_handle_t h;

    std::atomic<int> ecount = 0;

    for (int i = 1; i <= N; ++i)
    {
        s.submit(
            h,
            [&](sycl::queue, int i) {
                ecount += i;
                return sycl::event{};
            },
            i);
    }
    s.get_submission_group().wait();
    int count = ecount.load();
    EXPECT_EQ(0, count, "ERROR: scheduler did not execute all tasks exactly once\n");
    std::cout << "wait_on_scheduler empty list: OK\n";
    return 0;
}

int
test_submit_and_wait_on_sync()
{
    const int N = 100;
    oneapi::dpl::experimental::sycl_backend s;
    fake_selection_handle_t h;

    std::atomic<int> ecount = 0;

    for (int i = 1; i <= N; ++i)
    {
        auto w = s.submit(
            h,
            [&](sycl::queue, int i) {
                ecount += i;
                return sycl::event{};
            },
            i);
        w.wait();
        int count = ecount.load();
        EXPECT_EQ(i * (i + 1) / 2, count, "ERROR: scheduler did not execute all tasks exactly once\n");
    }
    std::cout << "wait_on_sync: OK\n";
    return 0;
}

int
test_submit_and_wait_on_sync_single_element()
{
    const int N = 1;
    oneapi::dpl::experimental::sycl_backend s;
    fake_selection_handle_t h;

    std::atomic<int> ecount = 0;

    for (int i = 1; i <= N; ++i)
    {
        auto w = s.submit(
            h,
            [&](sycl::queue, int i) {
                ecount += i;
                return sycl::event{};
            },
            i);
        w.wait();
        int count = ecount.load();
        EXPECT_EQ(1, count, "ERROR: scheduler did not execute all tasks exactly once\n");
    }
    std::cout << "wait_on_sync single element: OK\n";
    return 0;
}

int
test_submit_and_wait_on_sync_empty()
{
    const int N = 0;
    oneapi::dpl::experimental::sycl_backend s;
    fake_selection_handle_t h;

    std::atomic<int> ecount = 0;

    for (int i = 1; i <= N; ++i)
    {
        auto w = s.submit(
            h,
            [&](sycl::queue, int i) {
                ecount += i;
                return sycl::event{};
            },
            i);
        w.wait();
        int count = ecount.load();
        EXPECT_EQ(0, count, "ERROR: scheduler did not execute all tasks exactly once\n");
    }
    std::cout << "wait_on_sync empty list: OK\n";
    return 0;
}

int
test_properties()
{
    std::vector<sycl::queue> v;
    //= { sycl::queue(sycl::cpu_selector{}), sycl::queue(sycl::gpu_selector{}) };
    try
    {
        sycl::queue cpu_queue{sycl::cpu_selector_v};
        v.push_back(cpu_queue);
    }
    catch (const sycl::exception&)
    {
        std::cout << "SKIPPED: Unable to use cpu selector\n";
    }
    try
    {
        sycl::queue gpu_queue{sycl::gpu_selector_v};
        v.push_back(gpu_queue);
    }
    catch (const sycl::exception&)
    {
        std::cout << "SKIPPED: Unable to use gpu selector\n";
    }

    std::cout << "UNIVERSE SIZE " << v.size() << std::endl;
    oneapi::dpl::experimental::sycl_backend s(v);
    auto v2 = s.get_resources();
    auto v2s = v2.size();
    EXPECT_EQ(v2s, v.size(), "ERROR: reported universe and queried universe are not equal in size\n");
    for (int i = 0; i < v2s; ++i)
    {
        if (v[i] != oneapi::dpl::experimental::unwrap(v2[i]))
        {
            std::cout << "ERROR: reported universe and queried universe are not equal\n";
            return 1;
        }
    }
    std::cout << "properties: OK\n";
    return 0;
}
#endif

int
main()
{
    bool bProcessed = false;

#if TEST_DYNAMIC_SELECTION_AVAILABLE
    try
    {
        sycl::queue q;

        auto actual = test_cout();
        EXPECT_EQ(0, actual, "test_cout failed");
        actual = test_submit_and_wait_on_scheduler();
        actual = test_submit_and_wait_on_scheduler_single_element();
        actual = test_submit_and_wait_on_scheduler_empty();
        actual = test_submit_and_wait_on_sync();
        actual = test_submit_and_wait_on_sync_single_element();
        actual = test_submit_and_wait_on_sync_empty();
        actual = test_properties();

        bProcessed = true;
    }
    catch (const sycl::exception&)
    {
        std::cout << "SKIPPED: Unable to use sycl at all\n";
    }
#endif // TEST_DYNAMIC_SELECTION_AVAILABLE

    return TestUtils::done(bProcessed);
}
