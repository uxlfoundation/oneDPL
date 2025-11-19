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
launch_kernel(ResourceType& q, Adapter adapter, int* j, volatile float* v)
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

    TestUtils::usm_data_transfer<sycl::usm::alloc::shared, float> dt_helper_v(adapter(u[0]), 1000000);
    TestUtils::usm_data_transfer<sycl::usm::alloc::shared, int> dt_helper_j(adapter(u[0]), 1);

    //Making v volatile so the release build does not optimize it in the for loop below
    volatile float* v = dt_helper_v.get_data();
    int* j = dt_helper_j.get_data();


    my_policy_t p{u, adapter};
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

    TestUtils::usm_data_transfer<sycl::usm::alloc::shared, float> dt_helper_v(adapter(u[0]), 1000000);
    TestUtils::usm_data_transfer<sycl::usm::alloc::shared, int> dt_helper_j(adapter(u[0]), 1);

    //Making v volatile so the release build does not optimize it in the for loop below
    volatile float* v = dt_helper_v.get_data();
    int* j = dt_helper_j.get_data();


    my_policy_t p{u, adapter};
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
    TestUtils::usm_data_transfer<sycl::usm::alloc::shared, float> dt_helper_v(adapter(u[0]), 1000000);
    TestUtils::usm_data_transfer<sycl::usm::alloc::shared, int> dt_helper_j(adapter(u[0]), 1);

    //Making v volatile so the release build does not optimize it in the for loop below
    volatile float* v = dt_helper_v.get_data();
    int* j = dt_helper_j.get_data();

    my_policy_t p{u, adapter};
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


static inline void
build_auto_tune_universe(std::vector<sycl::queue>& u)
{
    auto prop_list = sycl::property_list{sycl::property::queue::enable_profiling()};

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
#if SYCL_EXT_ONEAPI_PROFILING_TAG
#if !ONEDPL_FPGA_DEVICE || !ONEDPL_FPGA_EMULATOR
        using policy_t =
            oneapi::dpl::experimental::auto_tune_policy<sycl::queue, oneapi::dpl::identity,
                                                        oneapi::dpl::experimental::default_backend<sycl::queue>>;
        std::vector<sycl::queue> u;
        build_auto_tune_universe(u);

        if (u.size() > 1)
        {

            std::cout << "\nRunning auto_tune tests for sycl::queue ...\n";
            EXPECT_EQ(0, (test_auto_initialization(u)), "");
            EXPECT_EQ(0, (test_default_universe_initialization<policy_t>(oneapi::dpl::identity{})), "");

            EXPECT_EQ(0, (test_auto_submit_wait_on_event<policy_t, class Kernel1>(u, 0, oneapi::dpl::identity{})), "");
            EXPECT_EQ(0, (test_auto_submit_wait_on_event<policy_t, class Kernel2>(u, 1, oneapi::dpl::identity{})), "");
            EXPECT_EQ(0, (test_auto_submit_wait_on_event<policy_t, class Kernel3>(u, 2, oneapi::dpl::identity{})), "");
            EXPECT_EQ(0, (test_auto_submit_wait_on_event<policy_t, class Kernel4>(u, 3, oneapi::dpl::identity{})), "");
            EXPECT_EQ(0, (test_auto_submit_wait_on_group<policy_t, class Kernel5>(u, 0, oneapi::dpl::identity{})), "");
            EXPECT_EQ(0, (test_auto_submit_wait_on_group<policy_t, class Kernel6>(u, 1, oneapi::dpl::identity{})), "");
            EXPECT_EQ(0, (test_auto_submit_wait_on_group<policy_t, class Kernel7>(u, 2, oneapi::dpl::identity{})), "");
            EXPECT_EQ(0, (test_auto_submit_wait_on_group<policy_t, class Kernel8>(u, 3, oneapi::dpl::identity{})), "");
            EXPECT_EQ(0, (test_auto_submit_and_wait<policy_t, class Kernel9>(u, 0, oneapi::dpl::identity{})), "");
            EXPECT_EQ(0, (test_auto_submit_and_wait<policy_t, class Kernel10>(u, 1, oneapi::dpl::identity{})), "");
            EXPECT_EQ(0, (test_auto_submit_and_wait<policy_t, class Kernel11>(u, 2, oneapi::dpl::identity{})), "");
            EXPECT_EQ(0, (test_auto_submit_and_wait<policy_t, class Kernel12>(u, 3, oneapi::dpl::identity{})), "");

            // Test with sycl::queue* resources and dereference adapter
            auto deref_op = [](auto pointer) { return *pointer; };
            using policy_pointer_t = oneapi::dpl::experimental::auto_tune_policy<
                sycl::queue*, decltype(deref_op),
                oneapi::dpl::experimental::default_backend<sycl::queue*, decltype(deref_op)>>;

            std::vector<sycl::queue*> u_ptrs;
            u_ptrs.reserve(u.size());
            for (auto& e : u)
            {
                u_ptrs.push_back(&e);
            }

            std::cout << "\nRunning auto_tune tests for sycl::queue* ...\n";
            EXPECT_EQ(0, (test_auto_submit_wait_on_event<policy_pointer_t, class Kernel37>(u_ptrs, 0, deref_op)), "");
            EXPECT_EQ(0, (test_auto_submit_wait_on_event<policy_pointer_t, class Kernel38>(u_ptrs, 1, deref_op)), "");
            EXPECT_EQ(0, (test_auto_submit_wait_on_event<policy_pointer_t, class Kernel39>(u_ptrs, 2, deref_op)), "");
            EXPECT_EQ(0, (test_auto_submit_wait_on_event<policy_pointer_t, class Kernel40>(u_ptrs, 3, deref_op)), "");
            EXPECT_EQ(0, (test_auto_submit_wait_on_group<policy_pointer_t, class Kernel41>(u_ptrs, 0, deref_op)), "");
            EXPECT_EQ(0, (test_auto_submit_wait_on_group<policy_pointer_t, class Kernel42>(u_ptrs, 1, deref_op)), "");
            EXPECT_EQ(0, (test_auto_submit_wait_on_group<policy_pointer_t, class Kernel43>(u_ptrs, 2, deref_op)), "");
            EXPECT_EQ(0, (test_auto_submit_wait_on_group<policy_pointer_t, class Kernel44>(u_ptrs, 3, deref_op)), "");
            EXPECT_EQ(0, (test_auto_submit_and_wait<policy_pointer_t, class Kernel45>(u_ptrs, 0, deref_op)), "");
            EXPECT_EQ(0, (test_auto_submit_and_wait<policy_pointer_t, class Kernel46>(u_ptrs, 1, deref_op)), "");
            EXPECT_EQ(0, (test_auto_submit_and_wait<policy_pointer_t, class Kernel47>(u_ptrs, 2, deref_op)), "");
            EXPECT_EQ(0, (test_auto_submit_and_wait<policy_pointer_t, class Kernel48>(u_ptrs, 3, deref_op)), "");

            //CTAD tests (testing policy construction without template arguments)
            //Template arguments types are deduced with CTAD
            sycl::queue q1(sycl::default_selector_v);
            sycl::queue q2(sycl::default_selector_v);

            //without resample time
            oneapi::dpl::experimental::auto_tune_policy p1{ {q1, q2} };
            oneapi::dpl::experimental::auto_tune_policy p2( {q1, q2} );

            oneapi::dpl::experimental::auto_tune_policy p3( {&q1, &q2}, deref_op );
            oneapi::dpl::experimental::auto_tune_policy p4{ {&q1, &q2}, deref_op };

            //with resample time
            oneapi::dpl::experimental::auto_tune_policy p5{ {q1, q2}, 1 };
            oneapi::dpl::experimental::auto_tune_policy p6( {q1, q2}, 1 );

            oneapi::dpl::experimental::auto_tune_policy p7( {&q1, &q2}, deref_op, 1 );
            oneapi::dpl::experimental::auto_tune_policy p8{ {&q1, &q2}, deref_op, 1 };

            bProcessed = true;
        }
        else
        {
            std::cout << "SKIPPED: Not enough valid devices to run auto_tune_policy tests\n";
        }
#endif // Devices available are CPU and GPU
#endif // SYCL_EXT_ONEAPI_PROFILING_TAG
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
