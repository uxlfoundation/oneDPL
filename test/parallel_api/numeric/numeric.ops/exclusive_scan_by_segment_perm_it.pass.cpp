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

class KernelName;
class KernelName1;
class KernelName2;

template <std::size_t N, typename TestValueType, typename USMAllocType>
void
test_exclusive_scan(sycl::queue q,
                    std::vector<TestValueType>& srcKeys,
                    std::vector<TestValueType>& srcVals,
                    std::vector<TestValueType>& expectedResults)
{
    constexpr auto alloc_type = USMAllocType::value;

    TestUtils::usm_data_transfer<alloc_type, TestValueType> dt_helper_keys(q, srcKeys.begin(), N);
    TestUtils::usm_data_transfer<alloc_type, TestValueType> dt_helper_vals(q, srcVals.begin(), N);
    TestUtils::usm_data_transfer<alloc_type, TestValueType> dt_helper_res(q, N);

    auto policy = TestUtils::make_device_policy<TestUtils::unique_kernel_name<
        TestUtils::unique_kernel_name<KernelName, 1>, TestUtils::uniq_kernel_index<alloc_type>()>>(q);

    oneapi::dpl::exclusive_scan_by_segment(
        policy,
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

template <std::size_t N, typename TestValueType, typename USMAllocType>
void
test_exclusive_scan(sycl::queue q,
                    std::vector<size_t>& perms,
                    std::vector<TestValueType>& srcKeys,
                    std::vector<TestValueType>& srcVals,
                    std::vector<TestValueType>& expectedResults)
{
    constexpr auto alloc_type = USMAllocType::value;

    TestUtils::usm_data_transfer<alloc_type, std::size_t> dt_helper_perm(q, perms.begin(), N);
    TestUtils::usm_data_transfer<alloc_type, TestValueType> dt_helper_keys(q, srcKeys.begin(), N);
    TestUtils::usm_data_transfer<alloc_type, TestValueType> dt_helper_vals(q, srcVals.begin(), N);
    TestUtils::usm_data_transfer<alloc_type, TestValueType> dt_helper_res(q, N);

    auto it_key_begin = oneapi::dpl::make_permutation_iterator(dt_helper_keys.get_data(), dt_helper_perm.get_data());
    auto it_key_end = it_key_begin + N;

    auto policy = TestUtils::make_device_policy<TestUtils::unique_kernel_name<
        TestUtils::unique_kernel_name<KernelName, 2>, TestUtils::uniq_kernel_index<alloc_type>()>>(q);

    oneapi::dpl::exclusive_scan_by_segment(
        policy,
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

template <std::size_t N, typename TestValueType>
void
test_exclusive_scan(sycl::queue q, std::vector<TestValueType>& srcKeys, std::vector<TestValueType>& srcVals,
                    std::vector<TestValueType>& expectedResults)
{
    std::vector<TestValueType> results(N);

    auto policy = TestUtils::make_device_policy<KernelName1>(q);

    oneapi::dpl::exclusive_scan_by_segment(policy,
                                           srcKeys.begin(),                 /* key begin */
                                           srcKeys.begin() + N,             /* key end */
                                           srcVals.begin(),                 /* input value begin */
                                           results.begin(),                 /* output value begin */
                                           0,                               /* init */
                                           std::equal_to<int>(), std::plus<int>());

    EXPECT_EQ_RANGES(expectedResults, results, "wrong effect from exclusive_scan_by_segment #1");
}

template <std::size_t N, typename TestValueType>
void
test_exclusive_scan(sycl::queue q, std::vector<size_t>& perms, std::vector<TestValueType>& srcKeys,
                    std::vector<TestValueType>& srcVals, std::vector<TestValueType>& expectedResults)
{
    auto it_key_begin = oneapi::dpl::make_permutation_iterator(srcKeys.begin(), perms.begin());
    auto it_key_end = it_key_begin + N;

    std::vector<TestValueType> results(N);

    auto policy = TestUtils::make_device_policy<KernelName2>(q);

    oneapi::dpl::exclusive_scan_by_segment(policy,
                                           it_key_begin,                    /* key begin */
                                           it_key_end,                      /* key end */
                                           srcVals.begin(),                 /* input value begin */
                                           results.begin(),                 /* output value begin */
                                           0,                               /* init */
                                           std::equal_to<int>(), std::plus<int>());

    EXPECT_EQ_RANGES(expectedResults, results, "wrong effect from exclusive_scan_by_segment #2");
}

#define _ONEDPL_PERM_BASE_ITERATOR_HOST_DEVICE_POL_SUPPORT 0

template <typename... Args>
void
test_exclusive_scan(sycl::queue q)
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
    test_exclusive_scan<N, TestValueType, Args...>(q, keys1, vals1, res1);

    // Keys: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
    // Vals: 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
    // Res:  0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    test_exclusive_scan<N, TestValueType, Args...>(q, keys2, vals2, res2);

    // Keys: 0, 0, 0, 0, 0, 1, 1, 1, 1, 1
    // Vals: 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
    // Res:  0, 1, 2, 3, 4, 0, 1, 2, 3, 4
    test_exclusive_scan<N, TestValueType, Args...>(q, keys1, vals1, res1);

#if _ONEDPL_PERM_BASE_ITERATOR_HOST_DEVICE_POL_SUPPORT
    // Perm: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
    // Keys: 0, 0, 0, 0, 0, 1, 1, 1, 1, 1
    // Vals: 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
    // Res:  0, 1, 2, 3, 4, 0, 1, 2, 3, 4
    test_exclusive_scan<N, TestValueType, Args...>(q, permutations1, keys1, vals1, res1);

    // Perm: 0, 0, 0, 0, 0, 1, 1, 1, 1, 1
    // Keys: 0, 0, 0, 0, 0, 1, 1, 1, 1, 1
    // Vals: 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
    // Res:  0, 1, 2, 3, 4, 5, 6, 7, 8, 9
    test_exclusive_scan<N, TestValueType, Args...>(q, permutations2, keys1, vals1, res3);

    // Perm: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
    // Keys: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
    // Vals: 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
    // Res:  0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    test_exclusive_scan<N, TestValueType, Args...>(q, permutations1, keys2, vals2, res2);

    // Perm: 0, 0, 0, 0, 0, 1, 1, 1, 1, 1
    // Keys: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
    // Vals: 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
    // Res:  0, 1, 2, 3, 4, 0, 1, 2, 3, 4
    test_exclusive_scan<N, TestValueType, Args...>(q, permutations2, keys2, vals2, res1);

    // Perm: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
    // Keys: 0, 0, 0, 0, 0, 1, 1, 1, 1, 1
    // Vals: 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
    // Res:  0, 1, 2, 3, 4, 0, 1, 2, 3, 4
    test_exclusive_scan<N, TestValueType, Args...>(q, permutations1, keys1, vals1, res1);

    // Perm: 0, 0, 0, 0, 0, 1, 1, 1, 1, 1
    // Keys: 0, 0, 0, 0, 0, 1, 1, 1, 1, 1
    // Vals: 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
    // Res:  0, 1, 2, 3, 4, 5, 6, 7, 8, 9
    test_exclusive_scan<N, TestValueType, Args...>(q, permutations2, keys1, vals1, res3);
#endif
}
#endif // TEST_DPCPP_BACKEND_PRESENT

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    sycl::queue q = TestUtils::get_test_queue();

    // Run tests for USM shared memory
    test_exclusive_scan<usm_alloc_type<sycl::usm::alloc::shared>>(q);

    // Run tests for USM device memory
    test_exclusive_scan<usm_alloc_type<sycl::usm::alloc::device>>(q);

    // Run tests for std::vector
    test_exclusive_scan(q);

#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
