// -*- C++ -*-
//===-- input_data_sweep_usm_allocator.pass.cpp ---------------------------===//
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
test_usm_shared_alloc(Policy&& exec, T trash, size_t n, const std::string& type_text)
{
    if (TestUtils::has_types_support<T>(exec.queue().get_device()))
    {
        //std::vector using usm shared allocator
        TestUtils::usm_data_transfer<sycl::usm::alloc::shared, T> copy_out(exec, n);
        oneapi::dpl::counting_iterator<int> counting(0);
        // usm_shared allocator std::vector
        sycl::usm_allocator<T, sycl::usm::alloc::shared> q_alloc{exec};
        std::vector<T, decltype(q_alloc)> shared_data_vec(n, q_alloc);
        //test all modes / wrappers

        //Only test as source iterator for permutation iterator if we can expect it to work
        // (if the vector implementation distinguishes its iterator for this type)
        wrap_recurse<
            __recurse, 0, /*__read =*/true, /*__reset_read=*/true, /*__write=*/true,
            /*__check_write=*/true, /*__usable_as_perm_map=*/true,
            /*__usable_as_perm_src=*/
            TestUtils::__vector_impl_distinguishes_usm_allocator_from_default_v<decltype(shared_data_vec.begin())>,
            /*__is_reversible=*/true>(std::forward<Policy>(exec), shared_data_vec.begin(), shared_data_vec.end(), counting,
                                      copy_out.get_data(), shared_data_vec.begin(), copy_out.get_data(), counting,
                                      trash, std::string("usm_shared_alloc_vector<") + type_text + std::string(">"));
    }
    else
    {
        TestUtils::unsupported_types_notifier(exec.queue().get_device());
    }
}

template <typename T, int __recurse, typename Policy>
void
test_usm_host_alloc(Policy&& exec, T trash, size_t n, const std::string& type_text)
{
    if (TestUtils::has_types_support<T>(exec.queue().get_device()))
    {
        //std::vector using usm host allocator
        TestUtils::usm_data_transfer<sycl::usm::alloc::shared, T> copy_out(exec, n);
        oneapi::dpl::counting_iterator<int> counting(0);
        // usm_host allocator std::vector
        sycl::usm_allocator<T, sycl::usm::alloc::host> q_alloc{exec};
        std::vector<T, decltype(q_alloc)> host_data_vec(n, q_alloc);
        //test all modes / wrappers

        //Only test as source iterator for permutation iterator if we can expect it to work
        // (if the vector implementation distiguishes its iterator for this type)
        wrap_recurse<
            __recurse, 0, /*__read =*/true, /*__reset_read=*/true, /*__write=*/true,
            /*__check_write=*/true, /*__usable_as_perm_map=*/true, /*__usable_as_perm_src=*/
            TestUtils::__vector_impl_distinguishes_usm_allocator_from_default_v<decltype(host_data_vec.begin())>,
            /*__is_reversible=*/true>(std::forward<Policy>(exec), host_data_vec.begin(), host_data_vec.end(), counting, copy_out.get_data(),
                                      host_data_vec.begin(), copy_out.get_data(), counting, trash,
                                      std::string("usm_host_alloc_vector<") + type_text + std::string(">"));
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
    test_usm_shared_alloc<float, 0>(CLONE_TEST_POLICY_IDX(exec, 0), -666.0f, n, "float");
    test_usm_shared_alloc<double, 0>(CLONE_TEST_POLICY_IDX(exec, 1), -666.0, n, "double");
    test_usm_shared_alloc<std::uint64_t, 0>(CLONE_TEST_POLICY_IDX(exec, 2), 999, n, "uint64_t");

#if !_PSTL_ICPX_FPGA_TEST_USM_VECTOR_ITERATOR_BROKEN
    // big recursion step: 1 and 2 layers of wrapping
    test_usm_shared_alloc<std::int32_t, 2>(CLONE_TEST_POLICY_IDX(exec, 3), -666, n, "int32_t");
#endif // !_PSTL_ICPX_FPGA_TEST_USM_VECTOR_ITERATOR_BROKEN

    //only use host alloc for int, it follows the same path as shared alloc
    test_usm_host_alloc<int, 0>(CLONE_TEST_POLICY_IDX(exec, 4), 666, n, "int");
}

#endif //TEST_DPCPP_BACKEND_PRESENT

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT

    auto policy = TestUtils::get_dpcpp_test_policy();
    test_impl(policy);

    TestUtils::check_compilation(policy, [](auto&& policy) { test_impl(std::forward<decltype(policy)>(policy)); });


#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
