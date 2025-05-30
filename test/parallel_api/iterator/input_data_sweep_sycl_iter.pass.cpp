// -*- C++ -*-
//===-- input_data_sweep_sycl_iter.pass.cpp -------------------------------===//
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
test(Policy&& policy, T trash, size_t n, const std::string& type_text)
{
    if (TestUtils::has_types_support<T>(policy.queue().get_device()))
    {
        TestUtils::usm_data_transfer<sycl::usm::alloc::shared, T> copy_out(policy.queue(), n);
        oneapi::dpl::counting_iterator<int> counting(0);
        // sycl iterator
        sycl::buffer<T> buf(n);
        //test all modes / wrappers
        wrap_recurse<__recurse, 0, /*__read =*/true, /*__reset_read=*/true, /*__write=*/true,
                     /*__check_write=*/true, /*__usable_as_perm_map=*/true, /*__usable_as_perm_src=*/true,
                     /*__is_reversible=*/false>(policy, oneapi::dpl::begin(buf), oneapi::dpl::end(buf), counting,
                                                copy_out.get_data(), oneapi::dpl::begin(buf), copy_out.get_data(),
                                                counting, trash,
                                                std::string("sycl_iterator<") + type_text + std::string(">"));
    }
    else
    {
        TestUtils::unsupported_types_notifier(policy.queue().get_device());
    }
}

#endif //TEST_DPCPP_BACKEND_PRESENT

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT

    constexpr size_t n = 10;

    auto policy = TestUtils::dpcpp_policy();

    auto policy1 = TestUtils::create_new_policy_idx<0>(policy);
    auto policy2 = TestUtils::create_new_policy_idx<1>(policy);
    auto policy3 = TestUtils::create_new_policy_idx<2>(policy);
    auto policy4 = TestUtils::create_new_policy_idx<3>(policy);

    // baseline with no wrapping
    test<float, 0>(policy1, -666.0f, n, "float");
    test<double, 0>(policy2, -666.0, n, "double");
    test<std::uint64_t, 0>(policy3, 999, n, "uint64_t");

    // big recursion step: 1 and 2 layers of wrapping
    test<std::int32_t, 2>(policy4, -666, n, "int32_t");

#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
