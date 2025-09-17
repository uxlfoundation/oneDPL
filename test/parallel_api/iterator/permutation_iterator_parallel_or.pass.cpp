// -*- C++ -*-
//===-- permutation_iterator_parallel_or.pass.cpp --------------------------===//
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

// dpl::is_heap, dpl::includes -> __parallel_or -> _parallel_find_or
DEFINE_TEST_PERM_IT(test_is_heap, PermItIndexTag)
{
    DEFINE_TEST_PERM_IT_CONSTRUCTOR(test_is_heap, 1.0f, 1.0f)

    template <typename TIterator>
    void generate_data(TIterator itBegin, TIterator itEnd, TestValueType initVal)
    {
        ::std::iota(itBegin, itEnd, initVal);
    }

    template <typename Size>
    struct TestImplementation
    {
        Size n;

        template <typename Policy, typename TPermutationIterator>
        void operator()(Policy&& exec, TPermutationIterator permItBegin, TPermutationIterator permItEnd) const
        {
            const auto testing_n = permItEnd - permItBegin;

            const auto resultIsHeap = dpl::is_heap(CLONE_TEST_POLICY(exec), permItBegin, permItEnd);
            wait_and_throw(exec);

            // Copy data back
            std::vector<TestValueType> expected(testing_n);
            dpl::copy(CLONE_TEST_POLICY(exec), permItBegin, permItEnd, expected.begin());
            wait_and_throw(exec);

            const auto expectedIsHeap = std::is_heap(expected.begin(), expected.end());
            EXPECT_EQ(expectedIsHeap, resultIsHeap, "Wrong result of dpl::is_heap");
        }
    };

    template <typename Policy, typename Iterator1, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 /*last1*/, Size n)
    {
        if constexpr (is_base_of_iterator_category_v<::std::random_access_iterator_tag, Iterator1>)
        {
            for (bool bCallMakeHeap : {false, true})
            {
                TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);     // source data for is_heap check
                const auto host_keys_ptr = host_keys.get();

                // Fill full source data set
                generate_data(host_keys_ptr, host_keys_ptr + n, TestValueType{});
                if (bCallMakeHeap)
                    ::std::make_heap(host_keys_ptr, host_keys_ptr + n);
                host_keys.update_data();

                test_through_permutation_iterator<Iterator1, Size, PermItIndexTag>{first1, n}(
                    CLONE_TEST_POLICY(exec), TestImplementation<Size>{n});
            }
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
    // dpl::is_heap, dpl::includes -> __parallel_or -> _parallel_find_or
    test1buffer<sycl::usm::alloc::shared, ValueType, test_is_heap<ValueType, PermItIndexTag>>();
    test1buffer<sycl::usm::alloc::device, ValueType, test_is_heap<ValueType, PermItIndexTag>>();
#endif // TEST_DPCPP_BACKEND_PRESENT
    if constexpr (!std::is_same_v<PermItIndexTag, perm_it_index_tags_usm_shared>)
    {
        // Run tests on <std::vector::iterator> + <all_host_policies>
        // dpl::is_heap, dpl::includes -> __parallel_or -> _parallel_find_or
        test_algo_one_sequence<ValueType, test_is_heap<ValueType, PermItIndexTag>>(kZeroOffset);
    }
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
