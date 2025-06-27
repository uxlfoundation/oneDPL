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

#include <vector>
#include <type_traits>

#include "support/utils.h"

#if TEST_DPCPP_BACKEND_PRESENT
#include "support/sycl_alloc_utils.h"

template <sycl::usm::alloc alloc_type>
using usm_alloc_type = ::std::integral_constant<sycl::usm::alloc, alloc_type>;

template <std::size_t idx>
class KernelName;

template <std::size_t N, typename TestValueType, typename USMAllocType, typename Policy>
void
test_exclusive_scan(Policy&& exec,
                    std::vector<TestValueType>& srcKeys,
                    std::vector<TestValueType>& srcVals,
                    std::vector<TestValueType>& expectedResults)
{
    constexpr auto alloc_type = USMAllocType::value;

    TestUtils::usm_data_transfer<alloc_type, TestValueType> dt_helper_keys(exec, srcKeys.begin(), N);
    TestUtils::usm_data_transfer<alloc_type, TestValueType> dt_helper_vals(exec, srcVals.begin(), N);
    TestUtils::usm_data_transfer<alloc_type, TestValueType> dt_helper_res (exec, N);

    using _NewKernelName = TestUtils::unique_kernel_name<
        TestUtils::unique_kernel_name<KernelName<0>, 1>,
        TestUtils::uniq_kernel_index<alloc_type>()>;

    oneapi::dpl::exclusive_scan_by_segment(
        CLONE_TEST_POLICY_NAME(exec, _NewKernelName),
        dt_helper_keys.get_data(),          /* key begin */
        dt_helper_keys.get_data() + N,      /* key end */
        dt_helper_vals.get_data(),          /* input value begin */
        dt_helper_res.get_data(),           /* output value begin */
        0,                                  /* init */
        std::equal_to<int>(), std::plus<int>());

    std::vector<TestValueType> results(N);
    dt_helper_res.retrieve_data(results.begin());

    EXPECT_EQ_RANGES(expectedResults, results, "wrong effect from exclusive_scan_by_segment #1");
}

template <std::size_t N, typename TestValueType, typename USMAllocType, typename Policy>
void
test_exclusive_scan(Policy&& exec,
                    std::vector<size_t>& perms,
                    std::vector<TestValueType>& srcKeys,
                    std::vector<TestValueType>& srcVals,
                    std::vector<TestValueType>& expectedResults)
{
    constexpr auto alloc_type = USMAllocType::value;

    TestUtils::usm_data_transfer<alloc_type, std::size_t>   dt_helper_perm(exec, perms.begin(), N);
    TestUtils::usm_data_transfer<alloc_type, TestValueType> dt_helper_keys(exec, srcKeys.begin(), N);
    TestUtils::usm_data_transfer<alloc_type, TestValueType> dt_helper_vals(exec, srcVals.begin(), N);
    TestUtils::usm_data_transfer<alloc_type, TestValueType> dt_helper_res (exec, N);

    auto it_key_begin = oneapi::dpl::make_permutation_iterator(dt_helper_keys.get_data(), dt_helper_perm.get_data());
    auto it_key_end = it_key_begin + N;

    using _NewKernelName = TestUtils::unique_kernel_name<
        TestUtils::unique_kernel_name<KernelName<0>, 2>,
        TestUtils::uniq_kernel_index<alloc_type>()>;

    oneapi::dpl::exclusive_scan_by_segment(
        CLONE_TEST_POLICY_NAME(exec, _NewKernelName),
        it_key_begin,               /* key begin */
        it_key_end,                 /* key end */
        dt_helper_vals.get_data(),  /* input value begin */
        dt_helper_res.get_data(),   /* output value begin */
        0,                          /* init */
        std::equal_to<int>(), std::plus<int>());

    std::vector<TestValueType> results(N);
    dt_helper_res.retrieve_data(results.begin());

    EXPECT_EQ_RANGES(expectedResults, results, "wrong effect from exclusive_scan_by_segment #2");
}

template <std::size_t N, typename TestValueType, typename Policy>
void
test_exclusive_scan(Policy&& exec,
                    std::vector<TestValueType>& srcKeys, std::vector<TestValueType>& srcVals,
                    std::vector<TestValueType>& expectedResults)
{
    std::vector<TestValueType> results(N);

    oneapi::dpl::exclusive_scan_by_segment(
        CLONE_TEST_POLICY_NAME(exec, KernelName<1>),
        srcKeys.begin(),                 /* key begin */
        srcKeys.begin() + N,             /* key end */
        srcVals.begin(),                 /* input value begin */
        results.begin(),                 /* output value begin */
        0,                               /* init */
        std::equal_to<int>(), std::plus<int>());

    EXPECT_EQ_RANGES(expectedResults, results, "wrong effect from exclusive_scan_by_segment #1");
}

template <std::size_t N, typename TestValueType, typename Policy>
void
test_exclusive_scan(Policy&& exec,
                    std::vector<size_t>& perms, std::vector<TestValueType>& srcKeys,
                    std::vector<TestValueType>& srcVals, std::vector<TestValueType>& expectedResults)
{
    auto it_key_begin = oneapi::dpl::make_permutation_iterator(srcKeys.begin(), perms.begin());
    auto it_key_end = it_key_begin + N;

    std::vector<TestValueType> results(N);

    oneapi::dpl::exclusive_scan_by_segment(
        CLONE_TEST_POLICY_NAME(exec, KernelName<2>),
        it_key_begin,                    /* key begin */
        it_key_end,                      /* key end */
        srcVals.begin(),                 /* input value begin */
        results.begin(),                 /* output value begin */
        0,                               /* init */
        std::equal_to<int>(), std::plus<int>());

    EXPECT_EQ_RANGES(expectedResults, results, "wrong effect from exclusive_scan_by_segment #2");
}

#define _ONEDPL_PERM_BASE_ITERATOR_HOST_DEVICE_POL_SUPPORT 0

template <typename Policy, typename... Args>
void
test_exclusive_scan(Policy&& exec)
{
    constexpr std::size_t N = 10;
    using TestValueType = int;

    std::vector<std::size_t> permutations1 = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<std::size_t> permutations2 = {0, 0, 0, 0, 0, 1, 1, 1, 1, 1};

    std::vector<TestValueType> keys1 = {0, 0, 0, 0, 0, 1, 1, 1, 1, 1};
    std::vector<TestValueType> vals1 = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    std::vector<TestValueType> res1 =  {0, 1, 2, 3, 4, 0, 1, 2, 3, 4};

    std::vector<TestValueType> keys2 = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<TestValueType> vals2 = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    std::vector<TestValueType> res2 =  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    std::vector<TestValueType> res3 = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

    assert(N == permutations1.size());
    assert(N == permutations2.size());
    assert(N == keys1.size());
    assert(N == vals1.size());
    assert(N == res1.size());
    assert(N == keys2.size());
    assert(N == vals2.size());
    assert(N == res2.size());

    // Keys: 0, 0, 0, 0, 0, 1, 1, 1, 1, 1
    // Vals: 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
    // Res:  0, 1, 2, 3, 4, 0, 1, 2, 3, 4
    test_exclusive_scan<N, TestValueType, Args...>(CLONE_TEST_POLICY(exec), keys1, vals1, res1);

    // Keys: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
    // Vals: 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
    // Res:  0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    test_exclusive_scan<N, TestValueType, Args...>(CLONE_TEST_POLICY(exec), keys2, vals2, res2);

    // Keys: 0, 0, 0, 0, 0, 1, 1, 1, 1, 1
    // Vals: 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
    // Res:  0, 1, 2, 3, 4, 0, 1, 2, 3, 4
    test_exclusive_scan<N, TestValueType, Args...>(CLONE_TEST_POLICY(exec), keys1, vals1, res1);

#if _ONEDPL_PERM_BASE_ITERATOR_HOST_DEVICE_POL_SUPPORT
    // Perm: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
    // Keys: 0, 0, 0, 0, 0, 1, 1, 1, 1, 1
    // Vals: 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
    // Res:  0, 1, 2, 3, 4, 0, 1, 2, 3, 4
    test_exclusive_scan<N, TestValueType, Args...>(CLONE_TEST_POLICY(exec), permutations1, keys1, vals1, res1);

    // Perm: 0, 0, 0, 0, 0, 1, 1, 1, 1, 1
    // Keys: 0, 0, 0, 0, 0, 1, 1, 1, 1, 1
    // Vals: 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
    // Res:  0, 1, 2, 3, 4, 5, 6, 7, 8, 9
    test_exclusive_scan<N, TestValueType, Args...>(CLONE_TEST_POLICY(exec), permutations2, keys1, vals1, res3);

    // Perm: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
    // Keys: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
    // Vals: 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
    // Res:  0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    test_exclusive_scan<N, TestValueType, Args...>(CLONE_TEST_POLICY(exec), permutations1, keys2, vals2, res2);

    // Perm: 0, 0, 0, 0, 0, 1, 1, 1, 1, 1
    // Keys: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
    // Vals: 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
    // Res:  0, 1, 2, 3, 4, 0, 1, 2, 3, 4
    test_exclusive_scan<N, TestValueType, Args...>(CLONE_TEST_POLICY(exec), permutations2, keys2, vals2, res1);

    // Perm: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
    // Keys: 0, 0, 0, 0, 0, 1, 1, 1, 1, 1
    // Vals: 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
    // Res:  0, 1, 2, 3, 4, 0, 1, 2, 3, 4
    test_exclusive_scan<N, TestValueType, Args...>(CLONE_TEST_POLICY(exec), permutations1, keys1, vals1, res1);

    // Perm: 0, 0, 0, 0, 0, 1, 1, 1, 1, 1
    // Keys: 0, 0, 0, 0, 0, 1, 1, 1, 1, 1
    // Vals: 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
    // Res:  0, 1, 2, 3, 4, 5, 6, 7, 8, 9
    test_exclusive_scan<N, TestValueType, Args...>(CLONE_TEST_POLICY(exec), permutations2, keys1, vals1, res3);
#endif
}

template <typename Policy>
void test_impl(Policy&& exec)
{
    // Run tests for USM shared/device memory
    test_exclusive_scan<decltype(CLONE_TEST_POLICY(exec)), usm_alloc_type<sycl::usm::alloc::shared>>(CLONE_TEST_POLICY(exec));
    test_exclusive_scan<decltype(CLONE_TEST_POLICY(exec)), usm_alloc_type<sycl::usm::alloc::device>>(CLONE_TEST_POLICY(exec));

    // Run tests for std::vector
    test_exclusive_scan(CLONE_TEST_POLICY(exec));
}
#endif // TEST_DPCPP_BACKEND_PRESENT

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
