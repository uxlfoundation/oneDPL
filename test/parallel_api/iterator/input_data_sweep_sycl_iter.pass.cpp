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
call_wrap_recurse(Policy&& exec, T trash, size_t n, const std::string& type_text)
{
    if (TestUtils::has_types_support<T>(exec.queue().get_device()))
    {
        TestUtils::usm_data_transfer<sycl::usm::alloc::shared, T> copy_out(exec, n);
        oneapi::dpl::counting_iterator<int> counting(0);
        // sycl iterator
        sycl::buffer<T> buf(n);
        //test all modes / wrappers
        wrap_recurse<__recurse, 0, /*__read =*/true, /*__reset_read=*/true, /*__write=*/true,
                     /*__check_write=*/true, /*__usable_as_perm_map=*/true, /*__usable_as_perm_src=*/true,
                     /*__is_reversible=*/false>(std::forward<Policy>(exec), oneapi::dpl::begin(buf), oneapi::dpl::end(buf), counting,
                                                copy_out.get_data(), oneapi::dpl::begin(buf), copy_out.get_data(),
                                                counting, trash,
                                                std::string("sycl_iterator<") + type_text + std::string(">"));
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
    call_wrap_recurse<float, 0>(CLONE_TEST_POLICY_IDX(exec, 0), -666.0f, n, "float");
    call_wrap_recurse<double, 0>(CLONE_TEST_POLICY_IDX(exec, 1), -666.0, n, "double");
    call_wrap_recurse<std::uint64_t, 0>(CLONE_TEST_POLICY_IDX(exec, 2), 999, n, "uint64_t");

    // big recursion step: 1 and 2 layers of wrapping
    call_wrap_recurse<std::int32_t, 2>(CLONE_TEST_POLICY_IDX(exec, 3), -666, n, "int32_t");
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
