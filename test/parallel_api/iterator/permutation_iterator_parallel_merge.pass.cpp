// -*- C++ -*-
//===-- permutation_iterator_parallel_merge.pass.cpp -----------------------===//
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

#include "permutation_iterator_common.h"

// dpl::merge, dpl::inplace_merge -> __parallel_merge
DEFINE_TEST_PERM_IT(test_merge, PermItIndexTag)
{
    DEFINE_TEST_PERM_IT_CONSTRUCTOR(test_merge, 2.0f, 0.65f)

    template <typename TIterator>
    void generate_data(TIterator itBegin, TIterator itEnd, TestValueType initVal)
    {
        ::std::iota(itBegin, itEnd, initVal);
    }

    template <typename Policy, typename Iterator1, typename Iterator2, typename Iterator3, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, Iterator2 last2, Iterator3 first3, Iterator3 last3, Size n)
    {
        if constexpr (is_base_of_iterator_category_v<::std::random_access_iterator_tag, Iterator1>)
        {
            auto exec1 = TestUtils::create_new_policy_idx<0>(exec);
            auto exec2 = TestUtils::create_new_policy_idx<1>(exec);
            auto exec3 = TestUtils::create_new_policy_idx<2>(exec);

            TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);                                 // source data(1) for merge
            TestDataTransfer<UDTKind::eVals, Size> host_vals(*this, n);                                 // source data(2) for merge
            TestDataTransfer<UDTKind::eRes,  Size> host_res (*this, ::std::distance(first3, last3));    // merge results

            const auto host_keys_ptr = host_keys.get();
            const auto host_vals_ptr = host_vals.get();
            const auto host_res_ptr  = host_res.get();

            // Fill full source data set
            generate_data(host_keys_ptr, host_keys_ptr + n, TestValueType{});
            generate_data(host_vals_ptr, host_vals_ptr + n, TestValueType{} + n / 2);
            ::std::fill(host_res_ptr, host_res_ptr + n, TestValueType{});

            // Update data
            host_keys.update_data();
            host_vals.update_data();
            host_res.update_data();

            assert(::std::distance(first3, last3) >= ::std::distance(first1, last1) + ::std::distance(first2, last2));

            test_through_permutation_iterator<Iterator1, Size, PermItIndexTag>{first1, n}(
                [&](auto permItBegin1, auto permItEnd1) // KSATODO move lambda out?
                {
                    const auto testing_n1 = permItEnd1 - permItBegin1;

                    //ensure list is sorted (not necessarily true after permutation)
                    dpl::sort(exec1, permItBegin1, permItEnd1);
                    wait_and_throw(exec1);

                    // Copy data back
                    std::vector<TestValueType> srcData1(testing_n1);
                    dpl::copy(exec1, permItBegin1, permItEnd1, srcData1.begin());
                    wait_and_throw(exec1);

                    test_through_permutation_iterator<Iterator2, Size, PermItIndexTag>{first2, n}(
                        [&](auto permItBegin2, auto permItEnd2) // KSATODO move lambda out?
                        {
                            const auto testing_n2 = permItEnd2 - permItBegin2;

                            //ensure list is sorted (not necessarily true after permutation)
                            dpl::sort(exec2, permItBegin2, permItEnd2);
                            wait_and_throw(exec2);

                            const auto resultEnd = dpl::merge(exec, permItBegin1, permItEnd1, permItBegin2, permItEnd2, first3);
                            wait_and_throw(exec);
                            const auto resultSize = resultEnd - first3;

                            // Copy data back
                            std::vector<TestValueType> srcData2(testing_n2);
                            dpl::copy(exec2, permItBegin2, permItEnd2, srcData2.begin());
                            wait_and_throw(exec2);

                            std::vector<TestValueType> mergedDataResult(resultSize);
                            dpl::copy(exec3, first3, resultEnd, mergedDataResult.begin());
                            wait_and_throw(exec3);

                            // Check results
                            std::vector<TestValueType> mergedDataExpected(testing_n1 + testing_n2);
                            auto expectedEnd = std::merge(srcData1.begin(), srcData1.end(), srcData2.begin(), srcData2.end(), mergedDataExpected.begin());
                            const auto expectedSize = expectedEnd - mergedDataExpected.begin();
                            EXPECT_EQ(expectedSize, resultSize, "Wrong size from dpl::merge");
                            EXPECT_EQ_N(mergedDataExpected.begin(), mergedDataResult.begin(), expectedSize, "Wrong result of dpl::merge");
                        });
                });
        }
    }
};

template <typename ValueType, typename PermItIndexTag>
void
run_algo_tests()
{
    constexpr ::std::size_t kZeroOffset = 0;

#if TEST_DPCPP_BACKEND_PRESENT
    // Run tests on <USM::shared, USM::device, sycl::buffer> + <all_hetero_policies>
    // dpl::merge, dpl::inplace_merge -> __parallel_merge
    test3buffers<sycl::usm::alloc::shared, ValueType, test_merge<ValueType, PermItIndexTag>>(2);
    test3buffers<sycl::usm::alloc::device, ValueType, test_merge<ValueType, PermItIndexTag>>(2);
#endif // TEST_DPCPP_BACKEND_PRESENT

    // Run tests on <std::vector::iterator> + <all_host_policies>
    // dpl::merge, dpl::inplace_merge -> __parallel_merge
    test_algo_three_sequences<ValueType, test_merge<ValueType, PermItIndexTag>>(2, kZeroOffset, kZeroOffset, kZeroOffset);
}

int
main()
{
    using ValueType = ::std::uint32_t;

#if TEST_DPCPP_BACKEND_PRESENT
    run_algo_tests<ValueType, perm_it_index_tags_usm_shared>();
#endif // TEST_DPCPP_BACKEND_PRESENT

    run_algo_tests<ValueType, perm_it_index_tags_counting>();
    run_algo_tests<ValueType, perm_it_index_tags_host>();
    run_algo_tests<ValueType, perm_it_index_tags_transform_iterator>();
    run_algo_tests<ValueType, perm_it_index_tags_callable_object>();

    return TestUtils::done();
}
