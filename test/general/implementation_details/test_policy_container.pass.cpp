// -*- C++ -*-
//===-- policy container.pass.cpp ------------------------------------------===//
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

#include <type_traits>
#include <utility>

#include "support/utils.h"

#if TEST_DPCPP_BACKEND_PRESENT
#include "support/utils_invoke.h"
#include "oneapi/dpl/pstl/hetero/dpcpp/execution_sycl_defs.h"
#endif

#if TEST_DPCPP_BACKEND_PRESENT

struct PassByValue{};
struct PassByConstReference{};
struct PassByMove{};

template <typename SourcePolicyKernelName, typename Policy, typename PassTag>
void test_policy_container(Policy&& exec, PassTag)
{
#if TEST_EXPLICIT_KERNEL_NAMES
    using ThisPolicyKernelName = oneapi::dpl::__internal::__policy_kernel_name<Policy>;
    static_assert(!std::is_same_v<SourcePolicyKernelName, ThisPolicyKernelName>, "Temporary test policy should have unique Kernel name");
#endif

    using DecayedPolicy = std::decay_t<decltype(exec)>;

    if constexpr (std::is_same_v<PassTag, PassByValue>)
    {
        using DecayedPolicyRefRef = DecayedPolicy&&;

        static_assert(std::is_same_v<decltype(exec), DecayedPolicyRefRef>, "Invalid test policy value category #0");
    }

    if constexpr (std::is_same_v<PassTag, PassByConstReference>)
    {
        using DecayedPolicyConstRef = const DecayedPolicy&;

        static_assert(std::is_same_v<decltype(exec), DecayedPolicyConstRef>, "Invalid test policy value category #1");
    }

    if constexpr (std::is_same_v<PassTag, PassByMove>)
    {
        using DecayedPolicyConstRef = DecayedPolicy&&;

        static_assert(std::is_same_v<decltype(exec), DecayedPolicyConstRef>, "Invalid test policy value category #2");
    }
}

template <typename Policy>
void test_pass_by_value(Policy policy)
{
    using SourcePolicyKernelName = oneapi::dpl::__internal::__policy_kernel_name<Policy>;

    test_policy_container<SourcePolicyKernelName>(CLONE_TEST_POLICY_IDX(policy, 0), PassByValue{});
}

template <typename Policy>
void test_pass_by_const_ref(const Policy& policy)
{
    using SourcePolicyKernelName = oneapi::dpl::__internal::__policy_kernel_name<Policy>;

    test_policy_container<SourcePolicyKernelName>(CLONE_TEST_POLICY_IDX(policy, 0), PassByConstReference{});
}

template <typename Policy>
void test_pass_by_rval(Policy&& exec)
{
    using SourcePolicyKernelName = oneapi::dpl::__internal::__policy_kernel_name<Policy>;

    test_policy_container<SourcePolicyKernelName>(CLONE_TEST_POLICY_IDX(exec, 0), PassByMove{});
}

#endif // TEST_DPCPP_BACKEND_PRESENT

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT

    {
        const auto policy = TestUtils::get_dpcpp_test_policy();
        test_pass_by_value(policy);
    }

    {
        // Save the policy in a const reference
        const auto& policy = TestUtils::get_dpcpp_test_policy();
        test_pass_by_const_ref(policy);
    }

    {
        auto policy = TestUtils::get_dpcpp_test_policy();
        test_pass_by_rval(std::move(policy));
    }

#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
