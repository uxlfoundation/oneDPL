// -*- C++ -*-
//===-- permutation_iterator_parallel_merge.pass.cpp -----------------------===//
//
// Copyright (C) Intel Corpor            //ensure list is sorted (not necessarily true after permutation)
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
#include "support/utils_invoke.h" // CLONE_TEST_POLICY_IDX

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

    template <typename Size, typename Iterator3, typename TPermutationIterator>
    struct TestImplementationLevel1
    {
        Size n;
        std::vector<TestValueType>& srcData1;

        Iterator3 first3;
        TPermutationIterator permItBegin1;
        TPermutationIterator permItEnd1;

        template <typename Policy, typename TPermutationIteratorArg>
        void operator()(Policy&& exec, TPermutationIteratorArg permItBegin2, TPermutationIteratorArg permItEnd2) const
        {
            const auto testing_n1 = permItEnd1 - permItBegin1;
            const auto testing_n2 = permItEnd2 - permItBegin2;

            //ensure list is sorted (not necessarily true after permutation)
            dpl::sort(CLONE_TEST_POLICY_IDX(exec, 0), permItBegin2, permItEnd2);
            wait_and_throw(exec);

            const auto resultEnd = dpl::merge(CLONE_TEST_POLICY_IDX(exec, 1), permItBegin1, permItEnd1, permItBegin2, permItEnd2, first3);
            wait_and_throw(exec);
            const auto resultSize = resultEnd - first3;

            // Copy data back
            std::vector<TestValueType> srcData2(testing_n2);
            dpl::copy(CLONE_TEST_POLICY_IDX(exec, 2), permItBegin2, permItEnd2, srcData2.begin());
            wait_and_throw(exec);

            std::vector<TestValueType> mergedDataResult(resultSize);
            dpl::copy(CLONE_TEST_POLICY_IDX(exec, 3), first3, resultEnd, mergedDataResult.begin());
            wait_and_throw(exec);

            // Print full sequences for debugging
            std::cout << "\n=== MERGE OPERATION DEBUG ===\n";
            std::cout << "Input sequence 1 (size=" << testing_n1 << "): ";
            for (size_t i = 0; i < srcData1.size(); ++i) {
                std::cout << srcData1[i] << " ";
            }
            std::cout << "\n";
            
            std::cout << "Input sequence 2 (size=" << testing_n2 << "): ";
            for (size_t i = 0; i < srcData2.size(); ++i) {
                std::cout << srcData2[i] << " ";
            }
            std::cout << "\n";

            // Check results
            std::vector<TestValueType> mergedDataExpected(testing_n1 + testing_n2);
            auto expectedEnd = std::merge(srcData1.begin(), srcData1.end(), srcData2.begin(), srcData2.end(), mergedDataExpected.begin());
            const auto expectedSize = expectedEnd - mergedDataExpected.begin();
            
            std::cout << "Expected result (size=" << expectedSize << "): ";
            for (size_t i = 0; i < expectedSize; ++i) {
                std::cout << mergedDataExpected[i] << " ";
            }
            std::cout << "\n";
            
            std::cout << "Actual result (size=" << resultSize << "): ";
            for (size_t i = 0; i < resultSize; ++i) {
                std::cout << mergedDataResult[i] << " ";
            }
            std::cout << "\n";
            
            if (expectedSize != resultSize) {
                std::cout << "SIZE MISMATCH: Expected " << expectedSize << ", got " << resultSize << "\n";
            } else {
                bool sequences_match = true;
                for (size_t i = 0; i < expectedSize; ++i) {
                    if (mergedDataExpected[i] != mergedDataResult[i]) {
                        std::cout << "VALUE MISMATCH at index " << i << ": Expected " 
                                  << mergedDataExpected[i] << ", got " << mergedDataResult[i] << "\n";
                        sequences_match = false;
                    }
                }
                if (sequences_match) {
                    std::cout << "✓ All values match!\n";
                }
            }
            std::cout << "=============================\n\n";
            
            EXPECT_EQ(expectedSize, resultSize, "Wrong size from dpl::merge");
            EXPECT_EQ_N(mergedDataExpected.begin(), mergedDataResult.begin(), expectedSize, "Wrong result of dpl::merge");
        }
    };

    template <typename Size, typename Iterator1, typename Iterator3>
    struct TestImplementationLevel0
    {
        Size n;
        Iterator1 first1;
        Iterator3 first3;

        template <typename Policy, typename TPermutationIterator>
        void operator()(Policy&& exec, TPermutationIterator permItBegin1, TPermutationIterator permItEnd1) const
        {
            const auto testing_n1 = permItEnd1 - permItBegin1;

            //ensure list is sorted (not necessarily true after permutation)
            dpl::sort(CLONE_TEST_POLICY(exec), permItBegin1, permItEnd1);
            wait_and_throw(exec);

            // Copy data back
            std::vector<TestValueType> srcData1(testing_n1);
            dpl::copy(CLONE_TEST_POLICY(exec), permItBegin1, permItEnd1, srcData1.begin());
            wait_and_throw(exec);

            std::cout << "=== AFTER PERMUTATION SORT 1 ===\n";
            std::cout << "Sorted sequence 1 (size=" << testing_n1 << "): ";
            for (size_t i = 0; i < srcData1.size(); ++i) {
                std::cout << srcData1[i] << " ";
            }
            std::cout << "\n";
            std::cout << "===============================\n\n";

            test_through_permutation_iterator<Iterator1, Size, PermItIndexTag>{first1, n}(
                std::forward<Policy>(exec), TestImplementationLevel1<Size, Iterator3, TPermutationIterator>{
                                                n, srcData1, first3, permItBegin1, permItEnd1});
        }
    };

    template <typename Policy, typename Size, typename Iterator1, typename Iterator2, typename Iterator3>
    void operator()(Policy&& exec, Iterator1 first1, [[maybe_unused]] Iterator1 last1,
                    [[maybe_unused]] Iterator2 first2, [[maybe_unused]] Iterator2 last2, Iterator3 first3,
                    Iterator3 last3, Size n)
    {
        if constexpr (is_base_of_iterator_category_v<std::random_access_iterator_tag, Iterator1>)
        {
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

            // Print initial data generation
            std::cout << "\n=== INITIAL DATA GENERATION ===\n";
            std::cout << "Full keys data (size=" << n << "): ";
            for (size_t i = 0; i < std::min(n, size_t(20)); ++i) {
                std::cout << host_keys_ptr[i] << " ";
            }
            if (n > 20) std::cout << "... (showing first 20)";
            std::cout << "\n";
            
            std::cout << "Full vals data (size=" << n << "): ";
            for (size_t i = 0; i < std::min(n, size_t(20)); ++i) {
                std::cout << host_vals_ptr[i] << " ";
            }
            if (n > 20) std::cout << "... (showing first 20)";
            std::cout << "\n";
            std::cout << "==============================\n\n";

            // Update data
            host_keys.update_data();
            host_vals.update_data();
            host_res.update_data();

            assert(::std::distance(first3, last3) >= ::std::distance(first1, last1) + ::std::distance(first2, last2));

            test_through_permutation_iterator<Iterator1, Size, PermItIndexTag>{first1, n}(
                std::forward<Policy>(exec), TestImplementationLevel0<Size, Iterator1, Iterator3>{n, first1, first3});
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
    //test3buffers<sycl::usm::alloc::device, ValueType, test_merge<ValueType, PermItIndexTag>>(2);
#endif // TEST_DPCPP_BACKEND_PRESENT

    // Run tests on <std::vector::iterator> + <all_host_policies>
    // dpl::merge, dpl::inplace_merge -> __parallel_merge
    //test_algo_three_sequences<ValueType, test_merge<ValueType, PermItIndexTag>>(2, kZeroOffset, kZeroOffset, kZeroOffset);
}

int
main()
{
    using ValueType = ::std::uint32_t;

#if TEST_DPCPP_BACKEND_PRESENT
    //run_algo_tests<ValueType, perm_it_index_tags_usm_shared>();
#endif // TEST_DPCPP_BACKEND_PRESENT

    //run_algo_tests<ValueType, perm_it_index_tags_counting>();
    run_algo_tests<ValueType, perm_it_index_tags_host>();
    //run_algo_tests<ValueType, perm_it_index_tags_transform_iterator>();
    //run_algo_tests<ValueType, perm_it_index_tags_callable_object>();

    return TestUtils::done();
}
