// -*- C++ -*-
//===-- permutation_iterator_parallel_stable_sort.pass.cpp -----------------===//
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

// test_sort : dpl::sort -> __parallel_stable_sort
// Requirements: only for random_access_iterator
DEFINE_TEST_PERM_IT(test_sort, PermItIndexTag)
{
    DEFINE_TEST_PERM_IT_CONSTRUCTOR(test_sort, 2.0f, 0.65f)

    template <typename TIterator, typename Size>
    void generate_data(TIterator itBegin, TIterator itEnd, Size n)
    {
        Size index = 0;
        for (auto it = itBegin; it != itEnd; ++it, ++index)
            *it = n - index;
    }

    template <typename TIterator>
    void check_results(TIterator itBegin, TIterator itEnd)
    {
        const auto result = std::is_sorted(itBegin, itEnd);
        EXPECT_TRUE(result, "Wrong sort data results");
    }

    template <typename Policy, typename Iterator1, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 last1, Size n)
    {
        if constexpr (is_base_of_iterator_category_v<::std::random_access_iterator_tag, Iterator1>)
        {
            TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);     // sorting data
            const auto host_keys_ptr = host_keys.get();

            test_through_permutation_iterator<Iterator1, Size, PermItIndexTag>{first1, n}(
                [&](auto permItBegin, auto permItEnd)
                {
                    const auto testing_n = permItEnd - permItBegin;

                    // Fill full source data set (not only values iterated by permutation iterator)
                    generate_data(host_keys_ptr, host_keys_ptr + n, n);
                    host_keys.update_data();

                    dpl::sort(exec, permItBegin, permItEnd);
                    wait_and_throw(exec);

                    // Copy data back
                    std::vector<TestValueType> resultTest(testing_n);
                    dpl::copy(exec, permItBegin, permItEnd, resultTest.begin());
                    wait_and_throw(exec);

                    // Check results
                    check_results(resultTest.begin(), resultTest.end());
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
    // test_sort : dpl::sort -> __parallel_stable_sort (only for random_access_iterator)
    test1buffer<sycl::usm::alloc::shared, ValueType, test_sort<ValueType, PermItIndexTag>>();
    test1buffer<sycl::usm::alloc::device, ValueType, test_sort<ValueType, PermItIndexTag>>();
#endif // TEST_DPCPP_BACKEND_PRESENT

    // Run tests on <std::vector::iterator> + <all_host_policies>
    // test_sort : dpl::sort -> __parallel_stable_sort (only for random_access_iterator)
    test_algo_one_sequence<ValueType, test_sort<ValueType, PermItIndexTag>>(kZeroOffset);
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
