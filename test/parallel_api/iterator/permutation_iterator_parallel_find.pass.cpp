// -*- C++ -*-
//===-- permutation_iterator_parallel_find.pass.cpp ------------------------===//
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

// dpl::find, dpl::find_if, dpl::find_if_not -> __parallel_find -> _parallel_find_or
DEFINE_TEST_PERM_IT(test_find, PermItIndexTag)
{
    DEFINE_TEST_PERM_IT_CONSTRUCTOR(test_find, 1.0f, 1.0f)

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

        template <typename TIterator>
        void
        operator()(TIterator permItBegin, TIterator permItEnd) const
        {
            const auto testing_n = permItEnd - permItBegin;
            if (testing_n >= 2)
            {
                // Get value to find: the second value
                std::vector<TestValueType> valueToFind(1);
                dpl::copy(exec, permItBegin + 1, permItBegin + 2, valueToFind.begin());
                wait_and_throw(exec);

                const auto result = dpl::find(exec, permItBegin, permItEnd, valueToFind[0]);
                wait_and_throw(exec);

                EXPECT_TRUE(result != permItEnd, "Wrong result of dpl::find");

                // Copy data back
                std::vector<TestValueType> foundVal(1);
                dpl::copy(exec, result, result + 1, foundVal.begin());
                wait_and_throw(exec);
                EXPECT_EQ(foundVal[0], valueToFind[0], "Incorrect value was found in dpl::find");
            }
        }
    };

    template <typename Policy, typename Iterator1, typename Size>
    void
    operator()(Policy&& exec, Iterator1 first1, Iterator1 /*last1*/, Size n)
    {
        assert(n > 0);

        if constexpr (is_base_of_iterator_category_v<::std::random_access_iterator_tag, Iterator1>)
        {
            TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);     // source data for find
            const auto host_keys_ptr = host_keys.get();

            // Fill full source data set
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
    // dpl::find, dpl::find_if, dpl::find_if_not -> __parallel_find -> _parallel_find_or
    test1buffer<sycl::usm::alloc::shared, ValueType, test_find<ValueType, PermItIndexTag>>();
    test1buffer<sycl::usm::alloc::device, ValueType, test_find<ValueType, PermItIndexTag>>();
#endif // TEST_DPCPP_BACKEND_PRESENT

    // Run tests on <std::vector::iterator> + <all_host_policies>
    // dpl::find, dpl::find_if, dpl::find_if_not -> __parallel_find -> _parallel_find_or
    test_algo_one_sequence<ValueType, test_find<ValueType, PermItIndexTag>>(kZeroOffset);
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
