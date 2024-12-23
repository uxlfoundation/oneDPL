// -*- C++ -*-
//===-- sycl_policy.pass.cpp ----------------------------------------------===//
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

#include "support/utils.h"

#include <iostream>
#include <vector>

#if TEST_DPCPP_BACKEND_PRESENT

template<typename Policy>
void test_policy_instance(const Policy& policy)
{
    auto __max_work_group_size = policy.queue().get_device().template get_info<sycl::info::device::max_work_group_size>();
    EXPECT_TRUE(__max_work_group_size > 0, "policy: wrong work group size");
    auto __max_compute_units = policy.queue().get_device().template get_info<sycl::info::device::max_compute_units>();
    EXPECT_TRUE(__max_compute_units > 0, "policy: wrong number of compute units");

    const int n = 10;
    static ::std::vector<int> a(n);

    ::std::fill(a.begin(), a.end(), 0);
    ::std::fill(policy, a.begin(), a.end(), -1);
#if _PSTL_SYCL_TEST_USM
    policy.queue().wait_and_throw();
#endif
    EXPECT_TRUE(::std::all_of(a.begin(), a.end(), [](int i) { return i == -1; }), "wrong result of ::std::fill with policy");
}
#endif // TEST_DPCPP_BACKEND_PRESENT

template<typename Policy>
constexpr void assert_is_execution_policy()
{
    static_assert(oneapi::dpl::is_execution_policy<Policy>::value, "wrong result for oneapi::dpl::is_execution_policy");
    static_assert(oneapi::dpl::is_execution_policy_v<Policy>, "wrong result for oneapi::dpl::is_execution_policy_v");
    static_assert(oneapi::dpl::execution::is_execution_policy<Policy>::value, "wrong result for oneapi::dpl::execution::is_execution_policy");
    static_assert(oneapi::dpl::execution::is_execution_policy_v<Policy>, "wrong result for oneapi::dpl::execution::is_execution_policy_v");
}

std::int32_t
main()
{
    using namespace oneapi::dpl::execution;
    assert_is_execution_policy<sequenced_policy>();
    assert_is_execution_policy<unsequenced_policy>();
    assert_is_execution_policy<parallel_policy>();
    assert_is_execution_policy<parallel_unsequenced_policy>();

    // Test that the policy is not decayed
    static_assert(!oneapi::dpl::is_execution_policy_v<sequenced_policy&&>, "wrong result for is_execution_policy_v<sequenced_policy&&>");

#if TEST_DPCPP_BACKEND_PRESENT
    auto q = sycl::queue{TestUtils::default_selector};

    assert_is_execution_policy<device_policy<class Kernel_0>>();

    test_policy_instance(dpcpp_default);

    // make_device_policy
    test_policy_instance(TestUtils::make_device_policy<class Kernel_11>(q));
#if TEST_LIBSYCL_VERSION && TEST_LIBSYCL_VERSION < 60000
    // make_device_policy requires a sycl::queue as an argument.
    // Currently, there is no implicit conversion (implicit syc::queue constructor by a device selector)
    // from a device selector to a queue.
    // The same test call with explicit queue creation we have below in line 78.
    test_policy_instance(TestUtils::make_device_policy<class Kernel_12>(TestUtils::default_selector));
#endif
    test_policy_instance(TestUtils::make_device_policy<class Kernel_13>(sycl::device{TestUtils::default_selector}));
    test_policy_instance(TestUtils::make_device_policy<class Kernel_14>(sycl::queue{TestUtils::default_selector, sycl::property::queue::in_order()}));
    test_policy_instance(TestUtils::make_device_policy<class Kernel_15>(dpcpp_default));
    // Special case: required to call make_device_policy directly from oneapi::dpl::execution namespace
    test_policy_instance(oneapi::dpl::execution::make_device_policy<class Kernel_16>());

    // device_policy
    EXPECT_TRUE(device_policy<class Kernel_1>(q).queue() == q, "wrong result for queue()");
    test_policy_instance(device_policy<class Kernel_21>(q));
    test_policy_instance(device_policy<class Kernel_22>(sycl::device{TestUtils::default_selector}));
    test_policy_instance(device_policy<class Kernel_23>(dpcpp_default));
    test_policy_instance(device_policy<class Kernel_24>(sycl::queue(dpcpp_default))); // conversion to sycl::queue
    test_policy_instance(device_policy<>{});
    class Kernel_25;
    static_assert(std::is_same_v<device_policy<Kernel_25>::kernel_name, Kernel_25>, "wrong result for kernel_name (device_policy)");

#if ONEDPL_FPGA_DEVICE
    assert_is_execution_policy<fpga_policy</*unroll_factor =*/ 1, class Kernel_0>>();
    test_policy_instance(dpcpp_fpga);

    // make_fpga_policy
    test_policy_instance(TestUtils::make_fpga_policy</*unroll_factor =*/ 1, class Kernel_31>(sycl::queue{TestUtils::default_selector}));
    test_policy_instance(TestUtils::make_fpga_policy</*unroll_factor =*/ 2, class Kernel_32>(sycl::device{TestUtils::default_selector}));
    test_policy_instance(TestUtils::make_fpga_policy</*unroll_factor =*/ 4, class Kernel_33>(dpcpp_fpga));
    // Special case: required to call make_fpga_policy directly from oneapi::dpl::execution namespace
    test_policy_instance(oneapi::dpl::execution::make_fpga_policy</*unroll_factor =*/ 8, class Kernel_34>());
    test_policy_instance(TestUtils::make_fpga_policy</*unroll_factor =*/ 16, class Kernel_35>(sycl::queue{TestUtils::default_selector}));

    // fpga_policy
    test_policy_instance(fpga_policy</*unroll_factor =*/ 1, class Kernel_41>(sycl::queue{TestUtils::default_selector}));
    test_policy_instance(fpga_policy</*unroll_factor =*/ 2, class Kernel_42>(sycl::device{TestUtils::default_selector}));
    test_policy_instance(fpga_policy</*unroll_factor =*/ 4, class Kernel_43>(dpcpp_fpga));
    test_policy_instance(fpga_policy</*unroll_factor =*/ 8, class Kernel_44>{});
    static_assert(std::is_same_v<fpga_policy</*unroll_factor =*/ 8, Kernel_25>::kernel_name, Kernel_25>, "wrong result for kernel_name (fpga_policy)");
    static_assert(fpga_policy</*unroll_factor =*/ 16, class Kernel_45>::unroll_factor == 16, "wrong unroll_factor");
#endif // ONEDPL_FPGA_DEVICE

#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}

