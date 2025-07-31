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
#include <iterator> // for std::reverse_iterator

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

template <typename It1, typename It2>
std::enable_if_t<std::is_same_v<bool, decltype(std::declval<It1>() == std::declval<It2>())>>
test_eq(It1 it1, It2 it2)
{
}

template <typename It1, typename It2>
std::enable_if_t<!std::is_same_v<bool, decltype(std::declval<It1>() == std::declval<It2>())>>
test_eq(It1 it1, It2 it2)
{
}

template <typename Policy>
void
test_diff_iterators(Policy&& exec)
{
    int* p1 = nullptr;
    bool* p2 = nullptr;

    //bool res = p1 == p2;
    //res = res;

    std::reverse_iterator<int*> it1(nullptr);
    std::reverse_iterator<bool*> it2(nullptr);

    //using rt = decltype(it1 == it2);
    test_eq(it1, it2);

#if 0

    const size_t size = 6;
    using SourceType      = bool;       // Data type of initial data
    using DestinationType = int;        // Data type of result data

    sycl::queue q = exec.queue();

    // Allocate USM shared memory for input (bool type) and output (int type)
    SourceType*      input           = sycl::malloc_shared<SourceType     >(size, q);
    DestinationType* result          = sycl::malloc_shared<DestinationType>(size, q);
    DestinationType* result_expected = sycl::malloc_shared<DestinationType>(size, q);

    // Initialize input data
    input[0] = true;
    input[1] = false;
    input[2] = true;
    input[3] = true;
    input[4] = false;
    input[5] = true;

    // Create reverse iterators from raw pointers
    // This creates reverse iterators with different underlying iterator types:
    // std::reverse_iterator<bool*> and std::reverse_iterator<int*>
    auto input_rbegin           = std::reverse_iterator<SourceType*>(input + size);
    auto input_rend             = std::reverse_iterator<SourceType*>(input);
    auto result_rbegin          = std::reverse_iterator<DestinationType*>(result + size);
    auto result_rbegin_expected = std::reverse_iterator<DestinationType*>(result_expected + size);

    constexpr DestinationType initial_value = 0;

    // Use exclusive_scan with reverse iterators to convert bool to int
    // This will scan from right to left (due to reverse iterators)
    // The initial value (0) will appear at the rightmost position
    //decltype(input_rbegin)::dummy;      // std::reverse_iterator<bool *>
    //decltype(input_rend)::dummy;        // std::reverse_iterator<bool *>
    //decltype(result_rbegin)::dummy;     // std::reverse_iterator<int *>
    oneapi::dpl::exclusive_scan(
        CLONE_TEST_POLICY(exec),            // Parallel execution policy
        input_rbegin,                       // Start of reversed input range
        input_rend,                         // End of reversed input range
        result_rbegin,                      // Start of reversed output range
        initial_value);                     // Initial value

    // Calculate expected result using serial exclusive_scan
    std::exclusive_scan(
        input_rbegin,                       // Start of reversed input range
        input_rend,                         // End of reversed input range
        result_rbegin_expected,             // Start of reversed output range
        initial_value);                     // Initial value

    EXPECT_EQ_N(result_expected, result, size, "wrong effect from exclusive_scan with reverse iterators");

    sycl::free(input, q);
    sycl::free(result, q);
    sycl::free(result_expected, q);
#endif
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
