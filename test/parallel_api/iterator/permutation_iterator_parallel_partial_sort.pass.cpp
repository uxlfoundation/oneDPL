// -*- C++ -*-
//===-- permutation_iterator_parallel_partial_sort.pass.cpp ---------------===//
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

// test_partial_sort : dpl::partial_sort -> __parallel_partial_sort
// Requirements: only for random_access_iterator
DEFINE_TEST_PERM_IT(test_partial_sort, PermItIndexTag)
{
    DEFINE_TEST_PERM_IT_CONSTRUCTOR(test_partial_sort, 1.0f, 1.0f)

    template <typename TIterator, typename Size>
    void generate_data(TIterator itBegin, TIterator itEnd, Size n)
    {
        Size index = 0;
        for (auto it = itBegin; it != itEnd; ++it, ++index)
            *it = n - index;
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

            // run at most 3 iters per n, 0 elements should be noop / cheap
            const auto partial_sorting_step = std::max(testing_n / 2, decltype(testing_n){1});
            for (std::size_t p = 0; p <= testing_n; p += partial_sorting_step)
            {
                dpl::partial_sort(exec, permItBegin, permItBegin + p, permItEnd);
                wait_and_throw(exec);

                // Copy data back
                std::vector<TestValueType> partialSortResult(p);
                dpl::copy(exec, permItBegin, permItBegin + p, partialSortResult.begin());
                wait_and_throw(exec);

                // Check results
                check_results(partialSortResult.begin(), partialSortResult.end());
            }
        }

        template <typename TIterator>
        void
        check_results(TIterator itBegin, TIterator itEnd) const
        {
            const auto result = std::is_sorted(oneapi::dpl::execution::par_unseq, itBegin, itEnd);
            EXPECT_TRUE(result, "Wrong partial_sort data results");
        }
    };

    template <typename Policy, typename Iterator1, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Size n)
    {
        if constexpr (is_base_of_iterator_category_v<::std::random_access_iterator_tag, Iterator1>)
        {
            TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);     // sorting data
            const auto host_keys_ptr = host_keys.get();

            // Fill full source data set (not only values iterated by permutation iterator)
            generate_data(host_keys_ptr, host_keys_ptr + n, n);
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
    // dpl::partial_sort -> __parallel_partial_sort (only for random_access_iterator)
    test1buffer<sycl::usm::alloc::shared, ValueType, test_partial_sort<ValueType, PermItIndexTag>>();
    test1buffer<sycl::usm::alloc::device, ValueType, test_partial_sort<ValueType, PermItIndexTag>>();
#endif // TEST_DPCPP_BACKEND_PRESENT

    // Run tests on <std::vector::iterator> + <all_host_policies>
    // dpl::partial_sort -> __parallel_partial_sort (only for random_access_iterator)
    test_algo_one_sequence<ValueType, test_partial_sort<ValueType, PermItIndexTag>>(kZeroOffset);
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
