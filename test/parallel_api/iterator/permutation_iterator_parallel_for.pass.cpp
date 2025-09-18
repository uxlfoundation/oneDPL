// -*- C++ -*-
//===-- permutation_iterator_parallel_for.pass.cpp -------------------------===//
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
#include "support/utils_invoke.h"

#include "permutation_iterator_common.h"

// dpl::transform -> __parallel_for
// Requirements: only for random_access_iterator
DEFINE_TEST_PERM_IT(test_transform, PermItIndexTag)
{
    DEFINE_TEST_PERM_IT_CONSTRUCTOR(test_transform, 1.0f, 1.0f)

    struct TransformOp
    {
        TestValueType operator()(TestValueType arg)
        {
            return arg * arg / 2;
        }
    };

    template <typename TIterator, typename Size>
    void generate_data(TIterator itBegin, TIterator itEnd, Size n)
    {
        Size index = 0;
        for (auto it = itBegin; it != itEnd; ++it, (void) ++index)
        {
            std::cout<<index<<".";
            *it = n - index;
        }
        std::cout<<std::endl;
    }

    template <typename Size, typename Iterator2>
    struct TestImplementation
    {
        Size n;
        TestDataTransfer<UDTKind::eVals, Size>& host_vals;      // reference to result data of transform
        Iterator2 first2;

        template <typename Policy, typename TPermutationIterator>
        void operator()(Policy&& exec, TPermutationIterator permItBegin, TPermutationIterator permItEnd) const
        {
            const auto testing_n = permItEnd - permItBegin;

            const auto host_vals_ptr = host_vals.get();
            clear_output_data(host_vals_ptr, host_vals_ptr + n);
            host_vals.update_data();
            std::cout<<"transform\n";
            auto itResultEnd = dpl::transform(CLONE_TEST_POLICY_IDX(exec, 0), permItBegin, permItEnd, first2, TransformOp{});
            wait_and_throw(exec);

            const auto resultSize = itResultEnd - first2;
            std::cout<<"copy back "<<testing_n<<"\n";
            // Copy data back
            std::vector<TestValueType> sourceData(testing_n);
            //dpl::copy(CLONE_TEST_POLICY_IDX(exec, 1), permItBegin, permItEnd, sourceData.begin());
            //wait_and_throw(exec);
	    std::cout<<"copy result\n";
            std::vector<TestValueType> transformedDataResult(testing_n);
            //dpl::copy(CLONE_TEST_POLICY_IDX(exec, 2), first2, itResultEnd, transformedDataResult.begin());
            //wait_and_throw(exec);
	    std::cout<<"verify\n";
            // Check results
            std::vector<TestValueType> transformedDataExpected(testing_n);
            //const auto itExpectedEnd = std::transform(sourceData.begin(), sourceData.end(), transformedDataExpected.begin(), TransformOp{});
            //const auto expectedSize = itExpectedEnd - transformedDataExpected.begin();
            //EXPECT_EQ(expectedSize, resultSize, "Wrong size from dpl::transform");
            //EXPECT_EQ_N(transformedDataExpected.begin(), transformedDataResult.begin(), expectedSize, "Wrong result of dpl::transform");
        }

        template <typename TIterator>
        void clear_output_data(TIterator itBegin, TIterator itEnd) const
        {
            std::fill(itBegin, itEnd, TestValueType{});
        }
    };

    template <typename Policy, typename Iterator1, typename Iterator2, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 /*last1*/, Iterator2 first2, Iterator2 /*last2*/, Size n)
    {
        if constexpr (is_base_of_iterator_category_v<::std::random_access_iterator_tag, Iterator1>)
        {
            TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);     // source data for transform
            TestDataTransfer<UDTKind::eVals, Size> host_vals(*this, n);     // result data of transform

            const auto host_keys_ptr = host_keys.get();
            std::cout<<"about to gen data for size "<<n<<std::endl;
            // Fill full source data set (not only values iterated by permutation iterator)
            generate_data(host_keys_ptr, host_keys_ptr + n, n);
            host_keys.update_data();

            test_through_permutation_iterator<Iterator1, Size, PermItIndexTag>{first1, n}(
                std::forward<Policy>(exec), TestImplementation<Size, Iterator2>{n, host_vals, first2});
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
    // dpl::transform -> __parallel_for (only for random_access_iterator)
    //std::cout<<"usm shared test2buffers\n";
    //test2buffers<sycl::usm::alloc::shared, ValueType, test_transform<ValueType, PermItIndexTag>>();
    //std::cout<<"usm device test2buffers\n";
    //test2buffers<sycl::usm::alloc::device, ValueType, test_transform<ValueType, PermItIndexTag>>();
#endif // TEST_DPCPP_BACKEND_PRESENT

    std::cout<<"host policies usm shared test_algo_two_sequences\n";
    // Run tests on <std::vector::iterator> + <all_host_policies>
    // dpl::transform -> __parallel_for (only for random_access_iterator)
    test_algo_two_sequences<ValueType, test_transform<ValueType, PermItIndexTag>>(kZeroOffset, kZeroOffset);
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
