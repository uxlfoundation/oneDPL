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

#include "support/utils.h"

#include _PSTL_TEST_HEADER(algorithm)
#include _PSTL_TEST_HEADER(iterator)
#include _PSTL_TEST_HEADER(functional)

#include <functional>
#include <iostream>

#if TEST_DPCPP_BACKEND_PRESENT
#include "support/sycl_alloc_utils.h"

class KernelName1;
class KernelName2;

template <sycl::usm::alloc alloc_type, typename KernelName, typename Policy>
void
test_with_usm(Policy&& exec)
{
    constexpr int n = 9;

    //data initialization
    int keys1  [n] = { 11, 11, 21, 20, 21, 21, 21, 37, 37 };
    int keys2  [n] = { 11, 11, 20, 20, 20, 21, 21, 37, 37 };
    int values1[n] = {  0,  1,  2,  3,  4,  5,  6,  7,  8 };
    int values2[n] = {  0,  1,  2,  3,  4,  5,  6,  7,  8 };
    int output_values1[n] = { };
    int output_values2[n] = { };

    // allocate USM memory and copying data to USM shared/device memory
    TestUtils::usm_data_transfer<alloc_type, int> dt_helper1(exec, keys1, n);
    TestUtils::usm_data_transfer<alloc_type, int> dt_helper2(exec, keys2, n);
    TestUtils::usm_data_transfer<alloc_type, int> dt_helper3(exec, values1, n);
    TestUtils::usm_data_transfer<alloc_type, int> dt_helper4(exec, values2, n);
    TestUtils::usm_data_transfer<alloc_type, int> dt_helper5(exec, output_values1, n);
    TestUtils::usm_data_transfer<alloc_type, int> dt_helper6(exec, output_values2, n);
    auto d_keys1          = dt_helper1.get_data();
    auto d_keys2          = dt_helper2.get_data();
    auto d_values1        = dt_helper3.get_data();
    auto d_values2        = dt_helper4.get_data();
    auto d_output_values1 = dt_helper5.get_data();
    auto d_output_values2 = dt_helper6.get_data();

    //make zip iterators
    auto begin_keys_in = oneapi::dpl::make_zip_iterator(d_keys1, d_keys2);
    auto end_keys_in   = oneapi::dpl::make_zip_iterator(d_keys1 + n, d_keys2 + n);
    auto begin_vals_in = oneapi::dpl::make_zip_iterator(d_values1, d_values2);
    auto begin_vals_out= oneapi::dpl::make_zip_iterator(d_output_values1, d_output_values2);

    //run inclusive_scan_by_segment algorithm 
    oneapi::dpl::inclusive_scan_by_segment(
        CLONE_TEST_POLICY_NAME(exec, KernelName),
        begin_keys_in, end_keys_in, begin_vals_in, begin_vals_out,
        std::equal_to<>(), TestUtils::TupleAddFunctor1());

    //retrieve result on the host and check the result
    dt_helper5.retrieve_data(output_values1);
    dt_helper6.retrieve_data(output_values2);

    // Expected output
    // {11, 11}: {0, 1}
    // {21, 20}: {2}
    // {20, 20}: {3}
    // {21, 20}: {4}
    // {21, 21}: {5, 11}
    // {37, 37}: {7, 15}
    const int exp_values1[n] = {0, 1, 2, 3, 4, 5, 11, 7, 15};
    const int exp_values2[n] = {0, 1, 2, 3, 4, 5, 11, 7, 15};
    EXPECT_EQ_N(exp_values1, output_values1, n, "wrong values1 from inclusive_scan_by_segment");
    EXPECT_EQ_N(exp_values2, output_values2, n, "wrong values2 from inclusive_scan_by_segment");
}

template <typename Policy>
void test_impl(Policy&& exec)
{
    // Run tests for USM shared/device memory
    test_with_usm<sycl::usm::alloc::shared, KernelName1>(CLONE_TEST_POLICY(exec));
    test_with_usm<sycl::usm::alloc::device, KernelName2>(CLONE_TEST_POLICY(exec));
}

#endif // TEST_DPCPP_BACKEND_PRESENT

int main()
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
