// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Test for sycl::inclusive_scan_over_group with init parameter
// This is a regression test for a compiler bug where inclusive_scan_over_group
// with an init parameter produces incorrect results on Intel GPUs in release builds.
// The bug manifests when scanning mask values (0 or 1) with std::plus operation.

#include "support/test_config.h"

#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>

#include "support/utils.h"

#include <random>

#if TEST_DPCPP_BACKEND_PRESENT

// Kernel names
class SyclScanKernel;
class CustomScanKernel;

// Test SYCL inclusive_scan_over_group with init parameter against custom implementation
template<typename Policy>
void
test_scan_with_init(Policy&& exec, std::size_t wg_size, std::size_t num_iterations)
{
    using namespace TestUtils;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> mask_distrib(0, 1);
    std::uniform_int_distribution<> init_distrib(0, 100);

    auto q = exec.queue();

    for (std::size_t iter = 0; iter < num_iterations; ++iter)
    {
        // Generate random mask pattern (0 or 1) and random init value
        std::vector<int> input(wg_size);
        for (std::size_t i = 0; i < wg_size; ++i)
        {
            input[i] = mask_distrib(gen);
        }
        int init_value = init_distrib(gen);

        std::vector<int> output_sycl(wg_size);
        std::vector<int> output_custom(wg_size);

        // Test SYCL inclusive_scan_over_group with init
        {
            sycl::buffer<int> in_buf(input.data(), sycl::range<1>(wg_size));
            sycl::buffer<int> out_buf(output_sycl.data(), sycl::range<1>(wg_size));

            q.submit([&](sycl::handler& h) {
                auto in = in_buf.get_access<sycl::access::mode::read>(h);
                auto out = out_buf.get_access<sycl::access::mode::write>(h);

                h.parallel_for<SyclScanKernel>(sycl::nd_range<1>(wg_size, wg_size), [=](sycl::nd_item<1> item) {
                    auto lid = item.get_local_id(0);
                    int value = in[lid];

                    // SYCL scan with init - this is the operation being tested
                    int result = sycl::inclusive_scan_over_group(
                        item.get_group(),
                        value,
                        std::plus<int>(),
                        init_value
                    );

                    out[lid] = result;
                });
            }).wait();
        }

        // Test custom implementation (reference from oneDPL unseq_backend_sycl.h #else branch)
        {
            sycl::buffer<int> in_buf(input.data(), sycl::range<1>(wg_size));
            sycl::buffer<int> out_buf(output_custom.data(), sycl::range<1>(wg_size));

            q.submit([&](sycl::handler& h) {
                auto in = in_buf.get_access<sycl::access::mode::read>(h);
                auto out = out_buf.get_access<sycl::access::mode::write>(h);
                sycl::local_accessor<int, 1> local_acc(sycl::range<1>(wg_size), h);

                h.parallel_for<CustomScanKernel>(sycl::nd_range<1>(wg_size, wg_size), [=](sycl::nd_item<1> item) {
                    auto lid = item.get_local_id(0);
                    int value = in[lid];

                    // Store in local memory
                    local_acc[lid] = value;
                    sycl::group_barrier(item.get_group());

                    // Custom scan implementation (from oneDPL #else branch)
                    int scan_result = value;
                    for (std::size_t i = lid; i > 0; --i)
                    {
                        scan_result = std::plus<int>()(local_acc[i - 1], scan_result);
                    }
                    sycl::group_barrier(item.get_group());

                    // Add init value
                    int result = std::plus<int>()(init_value, scan_result);
                    out[lid] = result;
                });
            }).wait();
        }

        // Calculate expected results
        std::vector<int> expected(wg_size);
        int cumsum = init_value;
        for (std::size_t i = 0; i < wg_size; ++i)
        {
            cumsum += input[i];
            expected[i] = cumsum;
        }

        // Verify SYCL implementation
        EXPECT_EQ_N(expected.data(), output_sycl.data(), wg_size,
                    "SYCL inclusive_scan_over_group with init produced incorrect results");

        // Verify custom implementation (should always pass)
        EXPECT_EQ_N(expected.data(), output_custom.data(), wg_size,
                    "Custom scan implementation produced incorrect results");
    }
}

#endif // TEST_DPCPP_BACKEND_PRESENT

std::int32_t
main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    // Test with different work group sizes
    constexpr std::size_t num_iterations = 100000;

    auto policy = TestUtils::get_dpcpp_test_policy();

    // Test with typical work group size
    test_scan_with_init(policy, 256, num_iterations);

    // Test with larger work group size
    auto q = policy.queue();
    auto max_wg_size = q.get_device().get_info<sycl::info::device::max_work_group_size>();
    if (max_wg_size >= 1024)
    {
        test_scan_with_init(policy, 1024, num_iterations);
    }

#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
