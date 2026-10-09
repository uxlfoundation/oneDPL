// -*- C++ -*-
//===-- cooperative_groups_estimate.cpp -----------------------------------===//
//
// Copyright (C) UXL Foundation Contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Implementation detail test for __get_num_cooperative_groups. A dummy kernel reserving a given amount of SLM is
// launched with the estimated number of work-groups and only completes if every group is resident at the same time.
// An estimate which is too large hangs the test.
//
// This tests asserts that decoupled lookback will not hang for any SLM size we requrest.

#include "../../support/test_config.h"

#include <oneapi/dpl/experimental/kernel_templates>

#include <algorithm>
#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <string>

#if __has_include(<sycl/sycl.hpp>)
#    include <sycl/sycl.hpp>
#else
#    include <CL/sycl.hpp>
#endif

#include "../../support/utils.h"

#if _ONEDPL_KT_COOPERATIVE_KERNELS_PRESENT

constexpr std::uint32_t kib = 1 << 10;

constexpr std::uint32_t max_slm_test_size = 384 * kib;

struct SpinBarrierKernel
{
    sycl::local_accessor<unsigned char, 1> slm;
    std::uint32_t* arrived_count;

    auto
    get(sycl::ext::oneapi::experimental::properties_tag) const
    {
        namespace syclex = sycl::ext::oneapi::experimental;
        return syclex::properties{syclex::work_group_progress<syclex::forward_progress_guarantee::concurrent,
                                                              syclex::execution_scope::root_group>};
    }

    void
    operator()(sycl::nd_item<1> item) const
    {
        const std::uint32_t lid = item.get_local_linear_id();
        const std::uint32_t work_group_size = item.get_local_range(0);
        const std::uint32_t num_groups = item.get_group_range(0);

        // Prevent the SLM allocation from being optimized away
        slm[lid] = 1;
        sycl::group_barrier(item.get_group());
        const std::uint32_t val = slm[(lid == 0) ? (work_group_size - 1) : (lid - 1)];

        if (lid == 0)
        {
            sycl::atomic_ref<std::uint32_t, sycl::memory_order::acq_rel, sycl::memory_scope::device,
                             sycl::access::address_space::global_space>
                arrived(*arrived_count);
            arrived.fetch_add(val);
            // Wait for all other work-group leaders to signal
            while (arrived.load(sycl::memory_order::acquire) != num_groups)
                ;
        }
    }
};

void
test_inter_work_group_barrier(sycl::queue& q, const sycl::kernel_bundle<sycl::bundle_state::executable>& bundle,
                              std::uint32_t work_group_size, std::uint32_t slm_size_bytes)
{
    namespace kt_impl = oneapi::dpl::experimental::kt::gpu::__impl;
    const std::uint32_t num_groups =
        kt_impl::__get_num_cooperative_groups(bundle.get_kernel<SpinBarrierKernel>(), q, work_group_size,
                                              std::numeric_limits<std::uint32_t>::max(), slm_size_bytes);

    std::uint32_t* arrived_count = sycl::malloc_device<std::uint32_t>(1, q);
    q.memset(arrived_count, 0, sizeof(std::uint32_t)).wait();
    q.submit([&](sycl::handler& cgh) {
         cgh.use_kernel_bundle(bundle);
         sycl::local_accessor<unsigned char, 1> slm(slm_size_bytes, cgh);
         cgh.parallel_for(sycl::nd_range<1>(num_groups * work_group_size, work_group_size),
                          SpinBarrierKernel{slm, arrived_count});
     }).wait();

    std::uint32_t host_arrived_count = 0;
    q.memcpy(&host_arrived_count, arrived_count, sizeof(std::uint32_t)).wait();
    sycl::free(arrived_count, q);

    const std::string config =
        "work-group size " + std::to_string(work_group_size) + ", SLM " + std::to_string(slm_size_bytes) + " bytes";
    EXPECT_EQ(num_groups, host_arrived_count, ("all work-groups must arrive: " + config).c_str());
}

#endif // _ONEDPL_KT_COOPERATIVE_KERNELS_PRESENT

int
main()
{
    bool run_test = false;
#if _ONEDPL_KT_COOPERATIVE_KERNELS_PRESENT
    sycl::queue q = TestUtils::get_test_queue();
    const sycl::device device = q.get_device();
    run_test = device.is_gpu();
    if (run_test)
    {
        try
        {
            sycl::kernel_bundle<sycl::bundle_state::executable> bundle =
                sycl::get_kernel_bundle<sycl::bundle_state::executable>(q.get_context(), {device},
                                                                        {sycl::get_kernel_id<SpinBarrierKernel>()});
            const std::uint32_t max_slm = device.get_info<sycl::info::device::local_mem_size>();
            const std::size_t max_work_group_size = device.get_info<sycl::info::device::max_work_group_size>();
            for (std::uint32_t work_group_size : {512u, 1024u})
            {
                if (work_group_size > max_work_group_size)
                    continue;
                for (std::uint32_t slm_size_bytes = work_group_size; slm_size_bytes <= max_slm_test_size;
                     slm_size_bytes += kib)
                {
                    if (slm_size_bytes > max_slm)
                        break;
                    test_inter_work_group_barrier(q, bundle, work_group_size, slm_size_bytes);
                }
            }
        }
        catch (const std::exception& exc)
        {
            std::cerr << "Exception: " << exc.what() << std::endl;
            return EXIT_FAILURE;
        }
    }
#endif

    return TestUtils::done(run_test);
}
