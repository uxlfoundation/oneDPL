// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Test for sycl::inclusive_scan_over_group with tuple pattern and local memory write-back
// This reproduces the pattern used in oneDPL's unique_copy where:
// 1. Tuples are loaded into local memory
// 2. A unary operation reads from local memory AND writes back to it
// 3. The result is then scanned with carry-over
//
// This pattern may trigger compiler bugs where the write-back to local memory
// interferes with the subsequent scan operation.

#include "support/test_config.h"

#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>

#include "support/utils.h"

#include <random>

#if TEST_DPCPP_BACKEND_PRESENT

// Kernel name
class TupleScanKernel;

// Struct to simulate tuple in local memory (like oneDPL's zip_view pattern)
struct DataMaskPair {
    int data;
    int mask;
};

// Mimics oneDPL's __create_mask which reads from local memory and writes back
struct CreateMask {
    template<typename Idx, typename LocalAcc>
    int operator()(Idx idx, LocalAcc& local_acc) const {
        // Read tuple from local memory
        auto& tuple_val = local_acc[idx];
        int data = tuple_val.data;

        // Generate mask value
        int mask = (data % 3 == 1 || data % 7 == 3) ? 1 : 0;

        // CRITICAL: Write mask back to local memory that was just read
        // This matches __create_mask doing: std::get<1>(__local_acc[__idx]) = __mask_value;
        tuple_val.mask = mask;

        return mask;
    }
};

// Test multi-iteration scan with tuple write-back pattern
template<typename Policy>
void
test_tuple_writeback_scan(Policy&& exec, std::size_t wg_size, std::size_t iters_per_wg, std::size_t num_tests)
{
    using namespace TestUtils;

#if 1
    const std::size_t elements_per_wg = wg_size * iters_per_wg;
    const std::size_t n_elements = 4*1024*1023 + 497;
    const std::size_t n_work_groups = n_elements / elements_per_wg + 1;
#else
    const std::size_t elements_per_wg = wg_size * iters_per_wg;
    const std::size_t n_work_groups = 4;
    const std::size_t n_elements = elements_per_wg * n_work_groups;
#endif

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> data_distrib(0, 1);

    auto q = exec.queue();

    for (std::size_t test = 0; test < num_tests; ++test)
    {
        // Generate random input data
        std::vector<int> input_data(n_elements);
        for (std::size_t i = 0; i < n_elements; ++i)
        {
            input_data[i] = data_distrib(gen);
        }

        std::vector<int> output_data(n_elements);

        // Run GPU kernel with tuple write-back pattern
        {
            sycl::buffer<int> data_buf(input_data.data(), sycl::range<1>(n_elements));
            sycl::buffer<int> output_buf(output_data.data(), sycl::range<1>(n_elements));

            q.submit([&](sycl::handler& cgh) {
                auto data_acc = data_buf.get_access<sycl::access::mode::read>(cgh);
                auto output_acc = output_buf.get_access<sycl::access::mode::write>(cgh);
                sycl::local_accessor<DataMaskPair, 1> local_acc(sycl::range<1>(wg_size), cgh);

                cgh.parallel_for<TupleScanKernel>(
                    sycl::nd_range<1>(n_work_groups * wg_size, wg_size),
                    [=](sycl::nd_item<1> item) {
                        auto group = item.get_group();
                        auto local_id = item.get_local_id(0);
                        auto group_id = item.get_group(0);

                        // Carry-over accumulator across iterations
                        int adder = 0;

                        std::size_t adjusted_global_id = local_id + elements_per_wg * group_id;

                        CreateMask create_mask_op;

                        // Multi-iteration loop with carry-over
                        for (std::size_t iter = 0; iter < iters_per_wg;
                             ++iter, adjusted_global_id += wg_size)
                        {
                            // Load data into local memory as tuple
                            if (adjusted_global_id < n_elements) {
                                local_acc[local_id].data = data_acc[adjusted_global_id];
                                local_acc[local_id].mask = 0;
                            } else {
                                local_acc[local_id].data = 0;
                                local_acc[local_id].mask = 0;  // identity
                            }

                            // CRITICAL PATTERN: Unary op reads from local memory AND writes back
                            // This matches oneDPL's __create_mask pattern:
                            //   _Tp __old_value = __unary_op(__local_id, __local_acc);
                            // where __unary_op modifies __local_acc
                            int mask_value = create_mask_op(local_id, local_acc);

                            // Barrier before scan
                            sycl::group_barrier(group);

                            // BUG LOCATION: inclusive_scan_over_group with init/adder
                            // after unary op has written back to local memory
                            int scan_result = sycl::inclusive_scan_over_group(
                                group,
                                mask_value,
                                std::plus<int>(),
                                adder
                            );

                            // Store result back to local memory
                            local_acc[local_id].mask = scan_result;

                            // Barrier after scan
                            sycl::group_barrier(group);

                            // Update adder for next iteration
                            adder = local_acc[wg_size - 1].mask;

                            // Write output
                            if (adjusted_global_id < n_elements)
                            {
                                output_acc[adjusted_global_id] = scan_result;
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
                if (global_idx < n_elements)
                {
                    auto data = input_data[global_idx];
                    int mask = (data % 3 == 1 || data % 7 == 3) ? 1 : 0;
                    cumsum += mask;
                    expected_data[global_idx] = cumsum;
                }
            }
        }

        // Verify results
        EXPECT_EQ_N(expected_data.data(), output_data.data(), n_elements,
                    "Tuple write-back scan pattern produced incorrect results");
    }
}

#endif // TEST_DPCPP_BACKEND_PRESENT

std::int32_t
main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    constexpr std::size_t num_tests = 100;
    std::size_t iters_per_wg = 4;

    auto policy = TestUtils::get_dpcpp_test_policy();
    auto q = policy.queue();

    std::size_t max_wg_size = q.get_device().get_info<sycl::info::device::max_work_group_size>();
    assert(max_wg_size >= 32);
    if (max_wg_size > 1024)
        max_wg_size = 1024;

    for (std::size_t wg_size = 32; wg_size <= max_wg_size; wg_size *= 2)
    {
        test_tuple_writeback_scan(policy, wg_size, iters_per_wg, num_tests);
    }

    test_tuple_writeback_scan(policy, max_wg_size, 2, num_tests);

#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
