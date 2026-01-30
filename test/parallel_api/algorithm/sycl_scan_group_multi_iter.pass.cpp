// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Test for sycl::inclusive_scan_over_group with multi-iteration carry-over
// This is a regression test for a compiler bug in the multi-iteration scan pattern
// used by oneDPL in unseq_backend_sycl.h where a work group processes multiple
// tiles of data with scan results carrying over between iterations.

#include "support/test_config.h"

#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>

#include "support/utils.h"

#include <random>

#if TEST_DPCPP_BACKEND_PRESENT

// Test multi-iteration scan pattern matching oneDPL's unseq_backend_sycl.h implementation
template<typename Policy>
void
test_multi_iter_scan(Policy&& exec, std::size_t wg_size, std::size_t iters_per_wg, std::size_t num_tests)
{
    using namespace TestUtils;

    const std::size_t elements_per_wg = wg_size * iters_per_wg;
    const std::size_t n_work_groups = 4;
    const std::size_t n_elements = elements_per_wg * n_work_groups;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> mask_distrib(0, 1);

    auto q = exec.queue();

    for (std::size_t test = 0; test < num_tests; ++test)
    {
        // Generate random mask pattern (0 or 1)
        std::vector<int> input_data(n_elements);
        for (std::size_t i = 0; i < n_elements; ++i)
        {
            input_data[i] = mask_distrib(gen);
        }

        std::vector<int> output_data(n_elements);

        // Run multi-iteration scan on GPU (matching oneDPL pattern)
        {
            sycl::buffer<int> input_buf(input_data.data(), sycl::range<1>(n_elements));
            sycl::buffer<int> output_buf(output_data.data(), sycl::range<1>(n_elements));

            q.submit([&](sycl::handler& cgh) {
                auto input_acc = input_buf.get_access<sycl::access::mode::read>(cgh);
                auto output_acc = output_buf.get_access<sycl::access::mode::write>(cgh);
                sycl::local_accessor<int, 1> local_acc(sycl::range<1>(wg_size), cgh);

                cgh.parallel_for(
                    sycl::nd_range<1>(n_work_groups * wg_size, wg_size),
                    [=](sycl::nd_item<1> item) {
                        auto group = item.get_group();
                        auto local_id = item.get_local_id(0);
                        auto group_id = item.get_group(0);

                        // Carry-over accumulator across iterations (identity for std::plus)
                        int adder = 0;

                        std::size_t adjusted_global_id = local_id + elements_per_wg * group_id;

                        // Multi-iteration loop with carry-over (matches unseq_backend_sycl.h:960-989)
                        for (std::size_t iter = 0; iter < iters_per_wg;
                             ++iter, adjusted_global_id += wg_size)
                        {
                            // Load into local memory
                            if (adjusted_global_id < n_elements)
                                local_acc[local_id] = input_acc[adjusted_global_id];
                            else
                                local_acc[local_id] = 0;  // identity

                            // Read value from local memory
                            int old_value = local_acc[local_id];

                            // Barrier before scan
                            sycl::group_barrier(group);

                            // BUG LOCATION: inclusive_scan_over_group with init/adder
                            local_acc[local_id] = sycl::inclusive_scan_over_group(
                                group,
                                old_value,
                                std::plus<int>(),
                                adder  // <-- Carry-over from previous iteration
                            );

                            // Barrier after scan
                            sycl::group_barrier(group);

                            // Update adder for next iteration
                            adder = local_acc[wg_size - 1];

                            // Store result
                            if (adjusted_global_id < n_elements)
                            {
                                output_acc[adjusted_global_id] = local_acc[local_id];
                            }
                        }
                    }
                );
            }).wait();
        }

        // Calculate expected results - cumulative sum within each work group
        std::vector<int> expected_data(n_elements);
        for (std::size_t wg = 0; wg < n_work_groups; ++wg)
        {
            int cumsum = 0;
            for (std::size_t i = 0; i < elements_per_wg; ++i)
            {
                std::size_t global_idx = wg * elements_per_wg + i;
                cumsum += input_data[global_idx];
                expected_data[global_idx] = cumsum;
            }
        }

        // Verify results
        EXPECT_EQ_N(expected_data.data(), output_data.data(), n_elements,
                    "Multi-iteration scan with carry-over produced incorrect results");
    }
}

#endif // TEST_DPCPP_BACKEND_PRESENT

std::int32_t
main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    constexpr std::size_t num_tests = 50;

    auto policy = TestUtils::get_dpcpp_test_policy();
    auto q = policy.queue();

    // Test with typical configuration: wg_size=256, 4 iterations per work group
    test_multi_iter_scan(policy, 256, 4, num_tests);

    // Test with larger work group if supported
    auto max_wg_size = q.get_device().get_info<sycl::info::device::max_work_group_size>();
    if (max_wg_size >= 1024)
    {
        // Single iteration (should pass even if multi-iter has issues)
        test_multi_iter_scan(policy, 1024, 1, num_tests);
    }

#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
