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
    oneapi::dpl::experimental::auto_tune_policy<sycl::queue, oneapi::dpl::identity,
                                                oneapi::dpl::experimental::default_backend<sycl::queue>>
        p{u};
    auto u2 = oneapi::dpl::experimental::get_resources(p);
    EXPECT_TRUE(std::equal(std::begin(u2), std::end(u2), std::begin(u)),
                "ERROR: provided resources and queried resources are not equal\n");

    // deferred initialization
    oneapi::dpl::experimental::auto_tune_policy<sycl::queue, oneapi::dpl::identity,
                                                oneapi::dpl::experimental::default_backend<sycl::queue>>
        p2{oneapi::dpl::experimental::deferred_initialization};
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
    EXPECT_TRUE(std::equal(std::begin(u3), std::end(u3), std::begin(u)),
                "ERROR: reported resources and queried resources are not equal after deferred initialization\n");

    std::cout << "initialization: OK\n" << std::flush;
    return 0;
}

template <typename KernelName, typename ResourceType, typename Adapter>
auto
launch_kernel(ResourceType& q, Adapter adapter, int* j, volatile double* v)
{
    return adapter(q).submit([=](sycl::handler& h) {
        h.parallel_for<KernelName>(
            1000000, [=](sycl::id<1> idx) {
                for (int j0 = 0; j0 < *j; ++j0)
                {
                    v[idx] += idx;
                }
            });
    });
}

template <typename Policy, typename KernelName, typename UniverseContainer, typename Adapter>
int
test_auto_submit_wait_on_event(UniverseContainer u, int best_resource, Adapter adapter)
{
    using my_policy_t = Policy;

    TestUtils::usm_data_transfer<sycl::usm::alloc::shared, double> dt_helper_v(adapter(u[0]), 1000000);
    TestUtils::usm_data_transfer<sycl::usm::alloc::shared, int> dt_helper_j(adapter(u[0]), 1);

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

                return launch_kernel<TestUtils::unique_kernel_name<KernelName, 1>>(q, adapter, j, v);
            });
        oneapi::dpl::experimental::wait(s);

        int count = ecount.load();
        EXPECT_EQ(i * (i + 1) / 2, count, "ERROR: scheduler did not execute all tasks exactly once\n");
    }
    EXPECT_TRUE(pass, "ERROR: did not select expected resources\n");

    std::cout << "submit and wait on event: OK\n";
    return 0;
}

template <typename Policy, typename KernelName, typename UniverseContainer, typename Adapter>
int
test_auto_submit_wait_on_group(UniverseContainer u, int best_resource, Adapter adapter)
{
    using my_policy_t = Policy;

    TestUtils::usm_data_transfer<sycl::usm::alloc::shared, double> dt_helper_v(adapter(u[0]), 1000000);
    TestUtils::usm_data_transfer<sycl::usm::alloc::shared, int> dt_helper_j(adapter(u[0]), 1);

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
                return launch_kernel<TestUtils::unique_kernel_name<KernelName, 1>>(q, adapter, j, v);
            });
        oneapi::dpl::experimental::wait(p.get_submission_group());

        int count = ecount.load();
        EXPECT_EQ(i * (i + 1) / 2, count, "ERROR: scheduler did not execute all tasks exactly once\n");
    }
    EXPECT_TRUE(pass, "ERROR: did not select expected resources\n");
    std::cout << "submit and wait on group: OK\n";
    return 0;
}

template <typename Policy, typename KernelName, typename UniverseContainer, typename Adapter>
int
test_auto_submit_and_wait(UniverseContainer u, int best_resource, Adapter adapter)
{
    using my_policy_t = Policy;

    // they are cpus so this is ok
    TestUtils::usm_data_transfer<sycl::usm::alloc::shared, double> dt_helper_v(adapter(u[0]), 1000000);
    TestUtils::usm_data_transfer<sycl::usm::alloc::shared, int> dt_helper_j(adapter(u[0]), 1);

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
            *j = 500;
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
                return launch_kernel<TestUtils::unique_kernel_name<KernelName, 1>>(q, adapter, j, v);
            });

        int count = ecount.load();
        EXPECT_EQ(i * (i + 1) / 2, count, "ERROR: scheduler did not execute all tasks exactly once\n");
    }
    EXPECT_TRUE(pass, "ERROR: did not select expected resources\n");
    std::cout << "submit_and_wait: OK\n";
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
        auto device_gpu1 = sycl::device(sycl::gpu_selector_v);
        sycl::queue gpu1_queue{device_gpu1, prop_list};
        u.push_back(gpu1_queue);
    }
    catch (const sycl::exception&)
    {
        std::cout << "SKIPPED: Unable to run with gpu_selector\n";
    }
    try
    {
        auto device_gpu2 = sycl::device(sycl::gpu_selector_v);
        sycl::queue gpu2_queue{device_gpu2, prop_list};
        u.push_back(gpu2_queue);
    }
    catch (const sycl::exception&)
    {
        std::cout << "SKIPPED: Unable to run with gpu_selector\n";
    }
    try
    {
        auto device_gpu3 = sycl::device(sycl::gpu_selector_v);
        sycl::queue gpu3_queue{device_gpu3, prop_list};
        u.push_back(gpu3_queue);
    }
    catch (const sycl::exception&)
    {
        std::cout << "SKIPPED: Unable to run with gpu_selector\n";
    }
    try
    {
        auto device_gpu4 = sycl::device(sycl::gpu_selector_v);
        sycl::queue gpu4_queue{device_gpu4, prop_list};
        u.push_back(gpu4_queue);
    }
    catch (const sycl::exception&)
    {
        std::cout << "SKIPPED: Unable to run with gpu_selector\n";
    }
}

#endif //TEST_DYNAMIC_SELECTION_AVAILABLE

int
main()
{
    bool bProcessed = false;

    try
    {
#if TEST_DYNAMIC_SELECTION_AVAILABLE
#if !ONEDPL_FPGA_DEVICE || !ONEDPL_FPGA_EMULATOR
        using policy_t =
            oneapi::dpl::experimental::auto_tune_policy<sycl::queue, oneapi::dpl::identity,
                                                        oneapi::dpl::experimental::default_backend<sycl::queue>>;
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

            std::cout << "\nRunning auto_tune tests for sycl::queue ...\n";
            EXPECT_EQ(0, (test_auto_initialization(u1)), "");
            EXPECT_EQ(0, (test_default_universe_initialization<policy_t>(oneapi::dpl::identity{})), "");

            EXPECT_EQ(0, (test_auto_submit_wait_on_event<policy_t, class Kernel1>(u1, 0, oneapi::dpl::identity{})), "");
            EXPECT_EQ(0, (test_auto_submit_wait_on_event<policy_t, class Kernel2>(u1, 1, oneapi::dpl::identity{})), "");
            EXPECT_EQ(0, (test_auto_submit_wait_on_event<policy_t, class Kernel3>(u1, 2, oneapi::dpl::identity{})), "");
            EXPECT_EQ(0, (test_auto_submit_wait_on_event<policy_t, class Kernel4>(u1, 3, oneapi::dpl::identity{})), "");
            EXPECT_EQ(0, (test_auto_submit_wait_on_group<policy_t, class Kernel5>(u1, 0, oneapi::dpl::identity{})), "");
            EXPECT_EQ(0, (test_auto_submit_wait_on_group<policy_t, class Kernel6>(u1, 1, oneapi::dpl::identity{})), "");
            EXPECT_EQ(0, (test_auto_submit_wait_on_group<policy_t, class Kernel7>(u1, 2, oneapi::dpl::identity{})), "");
            EXPECT_EQ(0, (test_auto_submit_wait_on_group<policy_t, class Kernel8>(u1, 3, oneapi::dpl::identity{})), "");
            EXPECT_EQ(0, (test_auto_submit_and_wait<policy_t, class Kernel9>(u1, 0, oneapi::dpl::identity{})), "");
            EXPECT_EQ(0, (test_auto_submit_and_wait<policy_t, class Kernel10>(u1, 1, oneapi::dpl::identity{})), "");
            EXPECT_EQ(0, (test_auto_submit_and_wait<policy_t, class Kernel11>(u1, 2, oneapi::dpl::identity{})), "");
            EXPECT_EQ(0, (test_auto_submit_and_wait<policy_t, class Kernel12>(u1, 3, oneapi::dpl::identity{})), "");
            // Use event profiling
            EXPECT_EQ(0, (test_auto_submit_wait_on_event<policy_t, class Kernel25>(u2, 0, oneapi::dpl::identity{})), "");
            EXPECT_EQ(0, (test_auto_submit_wait_on_event<policy_t, class Kernel26>(u2, 1, oneapi::dpl::identity{})), "");
            EXPECT_EQ(0, (test_auto_submit_wait_on_event<policy_t, class Kernel27>(u2, 2, oneapi::dpl::identity{})), "");
            EXPECT_EQ(0, (test_auto_submit_wait_on_event<policy_t, class Kernel28>(u2, 3, oneapi::dpl::identity{})), "");
            EXPECT_EQ(0, (test_auto_submit_wait_on_group<policy_t, class Kernel29>(u2, 0, oneapi::dpl::identity{})), "");
            EXPECT_EQ(0, (test_auto_submit_wait_on_group<policy_t, class Kernel30>(u2, 1, oneapi::dpl::identity{})), "");
            EXPECT_EQ(0, (test_auto_submit_wait_on_group<policy_t, class Kernel31>(u2, 2, oneapi::dpl::identity{})), "");
            EXPECT_EQ(0, (test_auto_submit_wait_on_group<policy_t, class Kernel32>(u2, 3, oneapi::dpl::identity{})), "");
            EXPECT_EQ(0, (test_auto_submit_and_wait<policy_t, class Kernel33>(u2, 0, oneapi::dpl::identity{})), "");
            EXPECT_EQ(0, (test_auto_submit_and_wait<policy_t, class Kernel34>(u2, 1, oneapi::dpl::identity{})), "");
            EXPECT_EQ(0, (test_auto_submit_and_wait<policy_t, class Kernel35>(u2, 2, oneapi::dpl::identity{})), "");
            EXPECT_EQ(0, (test_auto_submit_and_wait<policy_t, class Kernel36>(u2, 3, oneapi::dpl::identity{})), "");
            // Test with sycl::queue* resources and dereference adapter
            auto deref_op = [](auto pointer) { return *pointer; };
            using policy_pointer_t = oneapi::dpl::experimental::auto_tune_policy<
                sycl::queue*, decltype(deref_op),
                oneapi::dpl::experimental::default_backend<sycl::queue*, decltype(deref_op)>>;

            std::vector<sycl::queue*> u1_ptrs;
            u1_ptrs.reserve(u1.size());
            for (auto& e : u1)
            {
                u1_ptrs.push_back(&e);
            }

            std::vector<sycl::queue*> u2_ptrs;
            u2_ptrs.reserve(u2.size());
            for (auto& e : u2)
            {
                u2_ptrs.push_back(&e);
            }

            std::cout << "\nRunning auto_tune tests for sycl::queue* ...\n";
            EXPECT_EQ(0, (test_auto_submit_wait_on_event<policy_pointer_t, class Kernel37>(u1_ptrs, 0, deref_op)), "");
            EXPECT_EQ(0, (test_auto_submit_wait_on_event<policy_pointer_t, class Kernel38>(u1_ptrs, 1, deref_op)), "");
            EXPECT_EQ(0, (test_auto_submit_wait_on_event<policy_pointer_t, class Kernel39>(u1_ptrs, 2, deref_op)), "");
            EXPECT_EQ(0, (test_auto_submit_wait_on_event<policy_pointer_t, class Kernel40>(u1_ptrs, 3, deref_op)), "");
            EXPECT_EQ(0, (test_auto_submit_wait_on_group<policy_pointer_t, class Kernel41>(u1_ptrs, 0, deref_op)), "");
            EXPECT_EQ(0, (test_auto_submit_wait_on_group<policy_pointer_t, class Kernel42>(u1_ptrs, 1, deref_op)), "");
            EXPECT_EQ(0, (test_auto_submit_wait_on_group<policy_pointer_t, class Kernel43>(u1_ptrs, 2, deref_op)), "");
            EXPECT_EQ(0, (test_auto_submit_wait_on_group<policy_pointer_t, class Kernel44>(u1_ptrs, 3, deref_op)), "");
            EXPECT_EQ(0, (test_auto_submit_and_wait<policy_pointer_t, class Kernel45>(u1_ptrs, 0, deref_op)), "");
            EXPECT_EQ(0, (test_auto_submit_and_wait<policy_pointer_t, class Kernel46>(u1_ptrs, 1, deref_op)), "");
            EXPECT_EQ(0, (test_auto_submit_and_wait<policy_pointer_t, class Kernel47>(u1_ptrs, 2, deref_op)), "");
            EXPECT_EQ(0, (test_auto_submit_and_wait<policy_pointer_t, class Kernel48>(u1_ptrs, 3, deref_op)), "");
            // Use event profiling with pointers
            EXPECT_EQ(0, (test_auto_submit_wait_on_event<policy_pointer_t, class Kernel49>(u2_ptrs, 0, deref_op)), "");
            EXPECT_EQ(0, (test_auto_submit_wait_on_event<policy_pointer_t, class Kernel50>(u2_ptrs, 1, deref_op)), "");
            EXPECT_EQ(0, (test_auto_submit_wait_on_event<policy_pointer_t, class Kernel51>(u2_ptrs, 2, deref_op)), "");
            EXPECT_EQ(0, (test_auto_submit_wait_on_event<policy_pointer_t, class Kernel52>(u2_ptrs, 3, deref_op)), "");
            EXPECT_EQ(0, (test_auto_submit_wait_on_group<policy_pointer_t, class Kernel53>(u2_ptrs, 0, deref_op)), "");
            EXPECT_EQ(0, (test_auto_submit_wait_on_group<policy_pointer_t, class Kernel54>(u2_ptrs, 1, deref_op)), "");
            EXPECT_EQ(0, (test_auto_submit_wait_on_group<policy_pointer_t, class Kernel55>(u2_ptrs, 2, deref_op)), "");
            EXPECT_EQ(0, (test_auto_submit_wait_on_group<policy_pointer_t, class Kernel56>(u2_ptrs, 3, deref_op)), "");
            EXPECT_EQ(0, (test_auto_submit_and_wait<policy_pointer_t, class Kernel57>(u2_ptrs, 0, deref_op)), "");
            EXPECT_EQ(0, (test_auto_submit_and_wait<policy_pointer_t, class Kernel58>(u2_ptrs, 1, deref_op)), "");
            EXPECT_EQ(0, (test_auto_submit_and_wait<policy_pointer_t, class Kernel59>(u2_ptrs, 2, deref_op)), "");
            EXPECT_EQ(0, (test_auto_submit_and_wait<policy_pointer_t, class Kernel60>(u2_ptrs, 3, deref_op)), "");
            bProcessed = true;
        }
#endif // Devices available are CPU and GPU
#endif // TEST_DYNAMIC_SELECTION_AVAILABLE
    }
    catch (const std::exception& exc)
    {
        std::stringstream str;

        str << "Exception occurred";
        if (exc.what())
            str << " : " << exc.what();

        TestUtils::issue_error_message(str);
    }

    return TestUtils::done(bProcessed);
}
