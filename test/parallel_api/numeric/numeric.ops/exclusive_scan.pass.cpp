// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// This file incorporates work covered by the following copyright and permission
// notice:
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#include "support/test_config.h"

#include _PSTL_TEST_HEADER(execution)
#include _PSTL_TEST_HEADER(algorithm)
#include _PSTL_TEST_HEADER(numeric)

#include "support/utils.h"

#include <iostream>
#include <vector>

#include "support/scan_serial_impl.h"

#if TEST_DPCPP_BACKEND_PRESENT

#include "support/sycl_alloc_utils.h"

class TagCopy;

template <sycl::usm::alloc alloc_type, typename Policy>
void
test_with_usm(Policy&& exec, const std::size_t count)
{
    // Prepare source data
    std::vector<int> h_idx(count);
    for (int i = 0; i < count; i++)
        h_idx[i] = i + 1;

    // Copy source data to USM shared/device memory
    TestUtils::usm_data_transfer<alloc_type, int> dt_helper_h_idx(exec, std::begin(h_idx), std::end(h_idx));
    auto d_idx = dt_helper_h_idx.get_data();

    TestUtils::usm_data_transfer<alloc_type, int> dt_helper_h_val(exec, count);
    auto d_val = dt_helper_h_val.get_data();

    // Run dpl::exclusive_scan algorithm on USM shared-device memory
    using newKernelName = TestUtils::unique_kernel_name<TagCopy, TestUtils::uniq_kernel_index<alloc_type>()>;
    oneapi::dpl::exclusive_scan(CLONE_TEST_POLICY_NAME(exec, newKernelName), d_idx, d_idx + count, d_val, 0);

    // Copy results from USM shared/device memory to host
    std::vector<int> h_val(count);
    dt_helper_h_val.retrieve_data(h_val.begin());

    // Check results
    std::vector<int> h_sval_expected(count);
    exclusive_scan_serial(h_idx.begin(), h_idx.begin() + count, h_sval_expected.begin(), 0);

    EXPECT_EQ_N(h_sval_expected.begin(), h_val.begin(), count, "wrong effect from exclusive_scan");
}

template <sycl::usm::alloc alloc_type, typename Policy>
void
test_with_usm(Policy&& exec)
{
    for (::std::size_t n = 0; n <= TestUtils::max_n; n = n <= 16 ? n + 1 : size_t(3.1415 * n))
    {
        test_with_usm<alloc_type>(CLONE_TEST_POLICY(exec), n);
    }
}

template <typename Policy>
void
test_diff_iterators(Policy&& exec)
{
    constexpr std::size_t N = 6;

    sycl::queue q = exec.queue();
    
    // Allocate USM shared memory for input (bool type) and output (int type)
    bool* input = sycl::malloc_shared<bool>(N, q);
    int* result = sycl::malloc_shared<int>(N, q);
    
    // Initialize input data
    input[0] = true;
    input[1] = false;
    input[2] = true;
    input[3] = true;
    input[4] = false;
    input[5] = true;

    // Create reverse iterators
    auto input_rbegin = std::reverse_iterator<bool*>(input + N);
    auto input_rend = std::reverse_iterator<bool*>(input);

    constexpr int initial_value = 0;

    // Use exclusive_scan with reverse iterators to convert bool to int
    // This will scan from right to left (due to reverse iterators)
    // The initial value (0) will appear at the rightmost position
    auto result_rbegin = std::reverse_iterator<int*>(result + N);
    oneapi::dpl::exclusive_scan(
        std::forward<Policy>(exec),         // Parallel execution policy
        input_rbegin,                       // Start of reversed input range
        input_rend,                         // End of reversed input range
        result_rbegin,                      // Start of reversed output range
        initial_value                       // Initial value
    );

    // Calculate expected result using serial exclusive_scan
    std::vector<int> result_expected(N);
    auto result_rbegin_expected = result_expected.rbegin();
    std::exclusive_scan(
        input_rbegin,                       // Start of reversed input range
        input_rend,                         // End of reversed input range
        result_rbegin_expected,             // Start of reversed output range
        initial_value                       // Initial value
    );

    EXPECT_EQ_N(result_expected.data(), result, N, "wrong effect from exclusive_scan with reverse iterators");

    sycl::free(result, q);
    sycl::free(input, q);
}

template <typename Policy>
void test_impl(Policy&& exec)
{
    // Run tests for USM shared/device memory
    test_with_usm<sycl::usm::alloc::shared>(CLONE_TEST_POLICY(exec));
    test_with_usm<sycl::usm::alloc::device>(CLONE_TEST_POLICY(exec));

    test_diff_iterators(CLONE_TEST_POLICY(exec));
}
#endif // TEST_DPCPP_BACKEND_PRESENT

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT

    auto policy = TestUtils::get_dpcpp_test_policy();
    test_impl(policy);

#if TEST_CHECK_COMPILATION_WITH_DIFF_POLICY_VAL_CATEGORY
    TestUtils::check_compilation(policy, [](auto&& policy) { test_impl(std::forward<decltype(policy)>(policy)); });
#endif
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
