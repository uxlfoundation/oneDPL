// -*- C++ -*-
//===-- input_data_sweep_usm_device.pass.cpp ------------------------------===//
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

#include "support/utils.h"
#include _PSTL_TEST_HEADER(execution)
#include _PSTL_TEST_HEADER(algorithm)
#include _PSTL_TEST_HEADER(iterator)

#include "input_data_sweep.h"

#include "support/utils_invoke.h"

//This test is written without indirection from invoke_on_all_hetero_policies to make clear exactly which types
// are being tested, and to limit the number of types to be within reason.

#if TEST_DPCPP_BACKEND_PRESENT

template <typename T, int __recurse, typename Policy>
void
test_impl(Policy&& exec, T trash, size_t n, const std::string& type_text)
{
    if (TestUtils::has_types_support<T>(exec.queue().get_device()))
    {
        TestUtils::usm_data_transfer<sycl::usm::alloc::shared, T> copy_out(exec, n);
        oneapi::dpl::counting_iterator<int> counting(0);
        // usm_device
        TestUtils::usm_data_transfer<sycl::usm::alloc::device, T> device_data(exec, n);
        auto usm_device = device_data.get_data();
        //test all modes / wrappers
        wrap_recurse<__recurse, 0>(std::forward<Policy>(exec), usm_device, usm_device + n, counting, copy_out.get_data(), usm_device,
                                   copy_out.get_data(), counting, trash,
                                   std::string("usm_device<") + type_text + std::string(">"));
    }
    else
    {
        TestUtils::unsupported_types_notifier(exec.queue().get_device());
    }
}

template <typename Policy>
void
test_impl(Policy&& exec)
{
    constexpr size_t n = 10;

    // baseline with no wrapping
    test_impl<float, 0>(CLONE_TEST_POLICY_IDX(exec, 0), -666.0f, n, "float");
    test_impl<double, 0>(CLONE_TEST_POLICY_IDX(exec, 1), -666.0, n, "double");
    test_impl<std::uint64_t, 0>(CLONE_TEST_POLICY_IDX(exec, 2), 999, n, "uint64_t");

    // big recursion step: 1 and 2 layers of wrapping
    test_impl<std::int32_t, 2>(CLONE_TEST_POLICY_IDX(exec, 3), -666, n, "int32_t");
}

#endif //TEST_DPCPP_BACKEND_PRESENT

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
