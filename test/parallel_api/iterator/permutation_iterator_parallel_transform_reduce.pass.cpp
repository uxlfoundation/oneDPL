// -*- C++ -*-
//===-- permutation_iterator_parallel_transform_reduce.pass.cpp ------------===//
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

// dpl::reduce, dpl::transform_reduce -> __parallel_transform_reduce
// Requirements: only for random_access_iterator
DEFINE_TEST_PERM_IT(test_transform_reduce, PermItIndexTag)
{
    DEFINE_TEST_PERM_IT_CONSTRUCTOR(test_transform_reduce, 1.0f, 1.0f)

    template <typename TIterator>
    void generate_data(TIterator itBegin, TIterator itEnd, TestValueType initVal)
    {
        ::std::iota(itBegin, itEnd, initVal);
    }

    template <typename Policy, typename Size>
    struct TestImlementation
    {
        Policy exec;
        Size n;

        template <typename TPermutationIterator>
        void
        operator()(TPermutationIterator permItBegin, TPermutationIterator permItEnd) const
        {
            const auto testing_n = permItEnd - permItBegin;

            const auto result = dpl::transform_reduce(exec, permItBegin, permItEnd, TestValueType{},
                                                      std::plus<TestValueType>(), std::negate<TestValueType>());
            wait_and_throw(exec);

            // Copy data back
            std::vector<TestValueType> sourceData(testing_n);
            dpl::copy(exec, permItBegin, permItEnd, sourceData.begin());
            wait_and_throw(exec);

            const auto expected =
                TestUtils::transform_reduce_serial(sourceData.begin(), sourceData.end(), TestValueType{},
                                                   std::plus<TestValueType>(), std::negate<TestValueType>());
            EXPECT_EQ(expected, result, "Wrong result of dpl::transform_reduce");
        }
    };

    template <typename Policy, typename Iterator1, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 /*last1*/, Size n)
    {
        if constexpr (is_base_of_iterator_category_v<::std::random_access_iterator_tag, Iterator1>)
        {
            TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);     // source data for transform_reduce
            const auto host_keys_ptr = host_keys.get();

            // Fill full source data set (not only values iterated by permutation iterator)
            generate_data(host_keys_ptr, host_keys_ptr + n, TestValueType{});
            host_keys.update_data();

            test_through_permutation_iterator<Iterator1, Size, PermItIndexTag>{first1, n}(
                TestImlementation<Policy, Size>{exec, n});
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
    // dpl::reduce, dpl::transform_reduce -> __parallel_transform_reduce (only for random_access_iterator)
    test1buffer<sycl::usm::alloc::shared, ValueType, test_transform_reduce<ValueType, PermItIndexTag>>();
    test1buffer<sycl::usm::alloc::device, ValueType, test_transform_reduce<ValueType, PermItIndexTag>>();
#endif // TEST_DPCPP_BACKEND_PRESENT

    // Run tests on <std::vector::iterator> + <all_host_policies>
    // dpl::reduce, dpl::transform_reduce -> __parallel_transform_reduce (only for random_access_iterator)
    test_algo_one_sequence<ValueType, test_transform_reduce<ValueType, PermItIndexTag>>(kZeroOffset);
}

int
main()
{
    using ValueType = std::int32_t;

#if TEST_DPCPP_BACKEND_PRESENT
    run_algo_tests<ValueType, perm_it_index_tags_usm_shared>();
#endif // TEST_DPCPP_BACKEND_PRESENT

    run_algo_tests<ValueType, perm_it_index_tags_counting>();
    run_algo_tests<ValueType, perm_it_index_tags_host>();
    run_algo_tests<ValueType, perm_it_index_tags_transform_iterator>();
    run_algo_tests<ValueType, perm_it_index_tags_callable_object>();

    return TestUtils::done();
}
