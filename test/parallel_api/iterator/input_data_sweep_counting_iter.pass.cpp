// -*- C++ -*-
//===-- input_data_sweep_counting_iter.pass.cpp ---------------------------===//
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

#if TEST_DPCPP_BACKEND_PRESENT

//This test is written without indirection from invoke_on_all_hetero_policies to make clear exactly which types
// are being tested, and to limit the number of types to be within reason.

template <typename T, int __recurse, typename Policy>
void
test(Policy&& policy, T trash, size_t n, const std::string& type_text)
{
    if constexpr (std::is_integral_v<T>)
    {
        if (TestUtils::has_types_support<T>(policy.queue().get_device()))
        {

            TestUtils::usm_data_transfer<sycl::usm::alloc::shared, T> copy_out(policy.queue(), n);
            oneapi::dpl::counting_iterator<int> counting(0);
            oneapi::dpl::counting_iterator<T> my_counting(0);
            //counting_iterator
            wrap_recurse<__recurse, 0, /*__read =*/true, /*__reset_read=*/false, /*__write=*/false,
                         /*__check_write=*/false, /*__usable_as_perm_map=*/true, /*__usable_as_perm_src=*/true,
                         /*__is_reversible=*/true>(std::forward<Policy>(policy), my_counting, my_counting + n, counting, copy_out.get_data(),
                                                   my_counting, copy_out.get_data(), counting, trash,
                                                   std::string("counting_iterator<") + type_text + std::string(">"));
        }
        else
        {
            TestUtils::unsupported_types_notifier(policy.queue().get_device());
        }
    }
}

#endif //TEST_DPCPP_BACKEND_PRESENT

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT

    constexpr size_t n = 10;

    auto policy = TestUtils::get_dpcpp_test_policy();

    auto policy1 = TestUtils::create_new_policy_idx<0>(policy);
    auto policy2 = TestUtils::create_new_policy_idx<1>(policy);
    auto policy3 = TestUtils::create_new_policy_idx<2>(policy);
    auto policy4 = TestUtils::create_new_policy_idx<3>(policy);
    auto policy5 = TestUtils::create_new_policy_idx<4>(policy);

    // baseline with no wrapping
    test<float, 0>(policy1, -666.0f, n, "float");
    test<double, 0>(policy2, -666.0, n, "double");
    test<std::uint64_t, 0>(policy3, 999, n, "uint64_t");

    // big recursion step: 1 and 2 layers of wrapping
    test<std::int32_t, 2>(policy4, -666, n, "int32_t");

    // special case: discard iterator
    oneapi::dpl::counting_iterator<int> counting(0);
    oneapi::dpl::discard_iterator discard{};
    wrap_recurse<1, 0, /*__read =*/false, /*__reset_read=*/false, /*__write=*/true,
                 /*__check_write=*/false, /*__usable_as_perm_map=*/false, /*__usable_as_perm_src=*/true,
                 /*__is_reversible=*/true>(policy5, discard, discard + n, counting, discard, discard, discard, discard,
                                           -666, "discard_iterator");

#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
