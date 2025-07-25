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
#include <tuple>
#include <iterator>

#if TEST_DPCPP_BACKEND_PRESENT
#include "support/sycl_alloc_utils.h"

template <sycl::usm::alloc alloc_type, std::size_t KernelIdx, typename BinaryOp, typename Policy>
void
test_with_usm(Policy&& exec, BinaryOp binary_op)
{
    constexpr int n = 9;

    //data initialization
    int keys1  [n] = { 11, 11, 21, 20, 21, 21, 21, 37, 37 };
    int keys2  [n] = { 11, 11, 20, 20, 20, 21, 21, 37, 37 };
    int values1[n] = {  0,  1,  2,  3,  4,  5,  6,  7,  8 };
    int values2[n] = {  0,  1,  2,  3,  4,  5,  6,  7,  8 };
    int output_keys1  [n] = { };
    int output_keys2  [n] = { };
    int output_values1[n] = { };
    int output_values2[n] = { };

    // allocate USM memory and copying data to USM shared/device memory
    TestUtils::usm_data_transfer<alloc_type, int> dt_helper1(exec, keys1, n);
    TestUtils::usm_data_transfer<alloc_type, int> dt_helper2(exec, keys2, n);
    TestUtils::usm_data_transfer<alloc_type, int> dt_helper3(exec, values1, n);
    TestUtils::usm_data_transfer<alloc_type, int> dt_helper4(exec, values2, n);
    TestUtils::usm_data_transfer<alloc_type, int> dt_helper5(exec, output_keys1, n);
    TestUtils::usm_data_transfer<alloc_type, int> dt_helper6(exec, output_keys2, n);
    TestUtils::usm_data_transfer<alloc_type, int> dt_helper7(exec, output_values1, n);
    TestUtils::usm_data_transfer<alloc_type, int> dt_helper8(exec, output_values2, n);
    auto d_keys1          = dt_helper1.get_data();
    auto d_keys2          = dt_helper2.get_data();
    auto d_values1        = dt_helper3.get_data();
    auto d_values2        = dt_helper4.get_data();
    auto d_output_keys1   = dt_helper5.get_data();
    auto d_output_keys2   = dt_helper6.get_data();
    auto d_output_values1 = dt_helper7.get_data();
    auto d_output_values2 = dt_helper8.get_data();

    //make zip iterators
    auto begin_keys_in = oneapi::dpl::make_zip_iterator(d_keys1, d_keys2);
    auto end_keys_in   = oneapi::dpl::make_zip_iterator(d_keys1 + n, d_keys2 + n);
    auto begin_vals_in = oneapi::dpl::make_zip_iterator(d_values1, d_values2);
    auto begin_keys_out= oneapi::dpl::make_zip_iterator(d_output_keys1, d_output_keys2);
    auto begin_vals_out= oneapi::dpl::make_zip_iterator(d_output_values1, d_output_values2);

    //run reduce_by_segment algorithm
    using _NewKernelName = TestUtils::unique_kernel_name<BinaryOp, KernelIdx>;
    auto new_last = oneapi::dpl::reduce_by_segment(
        CLONE_TEST_POLICY_NAME(exec, _NewKernelName),
        begin_keys_in, end_keys_in, begin_vals_in, begin_keys_out, begin_vals_out,
        std::equal_to<>(), binary_op);

    //retrieve result on the host and check the result
    dt_helper5.retrieve_data(output_keys1);
    dt_helper6.retrieve_data(output_keys2);
    dt_helper7.retrieve_data(output_values1);
    dt_helper8.retrieve_data(output_values2);

//Dump
#if 0
    for(int i=0; i < n; i++) {
      std::cout << "{" << output_keys1[i] << ", " << output_keys2[i] << "}: "
                << "{" << output_values1[i] << ", " << output_values2[i] << "}" << std::endl;
    }
#endif

    // Expected output
    // {11, 11}: 1
    // {21, 20}: 2
    // {20, 20}: 3
    // {21, 20}: 4
    // {21, 21}: 11
    // {37, 37}: 15
    const int exp_keys1[n] = {11, 21, 20, 21, 21,37};
    const int exp_keys2[n] = {11, 20, 20, 20, 21, 37};
    const int exp_values1[n] = {1, 2, 3, 4, 11, 15};
    const int exp_values2[n] = {1, 2, 3, 4, 11, 15};
    EXPECT_EQ_N(exp_keys1, output_keys1, n, "wrong keys1 from reduce_by_segment");
    EXPECT_EQ_N(exp_keys2, output_keys2, n, "wrong keys2 from reduce_by_segment");
    EXPECT_EQ_N(exp_values1, output_values1, n, "wrong values1 from reduce_by_segment");
    EXPECT_EQ_N(exp_values2, output_values2, n, "wrong values2 from reduce_by_segment");
    EXPECT_EQ(std::distance(begin_keys_out, new_last.first), 6, "wrong number of keys from reduce_by_segment");
    EXPECT_EQ(std::distance(begin_vals_out, new_last.second), 6, "wrong number of values from reduce_by_segment");
}

template <std::size_t KernelIdx, typename Policy, typename BinaryOp>
void
test_zip_with_discard(Policy&& exec, BinaryOp binary_op)
{
    constexpr sycl::usm::alloc alloc_type = sycl::usm::alloc::device;

    constexpr int n = 5;

    //data initialization
    int keys1[n] = {1, 1, 2, 2, 3};
    int keys2[n] = {1, 1, 2, 2, 3};
    int values1[n] = {1, 1, 1, 1, 1};
    int values2[n] = {2, 2, 2, 2, 2};
    int output_keys[n] = {};
    int output_values[n] = {};

    // allocate USM memory and copying data to USM shared/device memory
    TestUtils::usm_data_transfer<alloc_type, int> dt_helper1(exec, keys1, n);
    TestUtils::usm_data_transfer<alloc_type, int> dt_helper2(exec, keys2, n);
    TestUtils::usm_data_transfer<alloc_type, int> dt_helper3(exec, values1, n);
    TestUtils::usm_data_transfer<alloc_type, int> dt_helper4(exec, values2, n);
    TestUtils::usm_data_transfer<alloc_type, int> dt_helper5(exec, output_keys, n);
    TestUtils::usm_data_transfer<alloc_type, int> dt_helper6(exec, output_values, n);
    auto d_keys1 = dt_helper1.get_data();
    auto d_keys2 = dt_helper2.get_data();
    auto d_values1 = dt_helper3.get_data();
    auto d_values2 = dt_helper4.get_data();
    auto d_output_keys = dt_helper5.get_data();
    auto d_output_values = dt_helper6.get_data();

    //make zip iterators
    auto begin_keys_in = oneapi::dpl::make_zip_iterator(d_keys1, d_keys2);
    auto end_keys_in = oneapi::dpl::make_zip_iterator(d_keys1 + n, d_keys2 + n);
    auto begin_vals_in = oneapi::dpl::make_zip_iterator(d_values1, d_values2);
    auto begin_keys_out = oneapi::dpl::make_zip_iterator(d_output_keys, oneapi::dpl::discard_iterator());
    auto begin_vals_out = oneapi::dpl::make_zip_iterator(oneapi::dpl::discard_iterator(), d_output_values);

    //run reduce_by_segment algorithm
    using _NewKernelName = TestUtils::unique_kernel_name<BinaryOp, KernelIdx>;
    auto new_last = oneapi::dpl::reduce_by_segment(
        CLONE_TEST_POLICY_NAME(exec, _NewKernelName),
        begin_keys_in, end_keys_in,
        begin_vals_in, begin_keys_out,
        begin_vals_out,
        std::equal_to<>(), binary_op);

    //retrieve result on the host and check the result
    dt_helper5.retrieve_data(output_keys);
    dt_helper6.retrieve_data(output_values);

    const int exp_keys[n] = {1, 2, 3};
    const int exp_values[n] = {4, 4, 2};
    EXPECT_EQ_N(exp_keys, output_keys, n, "wrong keys from reduce_by_segment");
    EXPECT_EQ_N(exp_values, output_values, n, "wrong values from reduce_by_segment");
    EXPECT_EQ(std::distance(begin_keys_out, new_last.first), 3, "wrong number of keys from reduce_by_segment");
    EXPECT_EQ(std::distance(begin_vals_out, new_last.second), 3, "wrong number of values from reduce_by_segment");
}

template <typename Policy, typename BinaryOp>
void test_with_op(Policy&& exec, BinaryOp binary_op)
{
    // Run tests for USM shared/device memory
    test_with_usm<sycl::usm::alloc::shared, 0>(CLONE_TEST_POLICY(exec), binary_op);
    test_with_usm<sycl::usm::alloc::device, 1>(CLONE_TEST_POLICY(exec), binary_op);

    test_zip_with_discard<2>(CLONE_TEST_POLICY(exec), binary_op);
}

template <typename Policy>
void test_impl(Policy&& exec)
{
    test_with_op(CLONE_TEST_POLICY(exec), TestUtils::TupleAddFunctor1{});
    test_with_op(CLONE_TEST_POLICY(exec), TestUtils::TupleAddFunctor2{});
}

#endif // TEST_DPCPP_BACKEND_PRESENT

//The code below for test a call of reduce_by_segment with zip iterators was kept "as is", as an example reported by a user; just "memory deallocation" added.
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
