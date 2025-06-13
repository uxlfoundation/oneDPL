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

template <typename Policy, sycl::usm::alloc alloc_type>
void
test_with_usm(Policy&& policy, const std::size_t count)
{
    // Prepare source data
    std::vector<int> h_idx(count);
    for (int i = 0; i < count; i++)
        h_idx[i] = i + 1;

    // Copy source data to USM shared/device memory
    TestUtils::usm_data_transfer<alloc_type, int> dt_helper_h_idx(policy.queue(), std::begin(h_idx), std::end(h_idx));
    auto d_idx = dt_helper_h_idx.get_data();

    TestUtils::usm_data_transfer<alloc_type, int> dt_helper_h_val(policy.queue(), count);
    auto d_val = dt_helper_h_val.get_data();

    // Run dpl::exclusive_scan algorithm on USM shared-device memory
    using newKernelName = TestUtils::unique_kernel_name<class copy, TestUtils::uniq_kernel_index<alloc_type>()>>;
    oneapi::dpl::exclusive_scan(CREATE_POLICY_WITH_NAME(policy, newKernelName), d_idx, d_idx + count, d_val, 0);

    // Copy results from USM shared/device memory to host
    std::vector<int> h_val(count);
    dt_helper_h_val.retrieve_data(h_val.begin());

    // Check results
    std::vector<int> h_sval_expected(count);
    exclusive_scan_serial(h_idx.begin(), h_idx.begin() + count, h_sval_expected.begin(), 0);

    EXPECT_EQ_N(h_sval_expected.begin(), h_val.begin(), count, "wrong effect from exclusive_scan");
}

template <typename Policy, sycl::usm::alloc alloc_type>
void
test_with_usm(Policy&& policy)
{
    for (::std::size_t n = 0; n <= TestUtils::max_n; n = n <= 16 ? n + 1 : size_t(3.1415 * n))
    {
        test_with_usm<alloc_type>(CLONE_NEW_POLICY(policy), n);
    }
}

template <typename Policy>
void test_impl(Policy&& policy)
{
    // Run tests for USM shared/device memory
    test_with_usm<sycl::usm::alloc::shared>(policy);
    test_with_usm<sycl::usm::alloc::device>(policy);
}
#endif // TEST_DPCPP_BACKEND_PRESENT

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT

    auto policy = TestUtils::get_test_dpcpp_policy();
    test_impl(policy);

    TestUtils::check_compile([](auto&& policy) { test_impl(std::forward<decltype(policy)>(policy)); });

#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
