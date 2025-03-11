// -*- C++ -*-
//===-- exclusive_scan_by_segment.pass.cpp ------------------------------------===//
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

#include "oneapi/dpl/execution"
#include "oneapi/dpl/algorithm"
#include "oneapi/dpl/iterator"

#include "support/utils.h"
#include "support/scan_serial_impl.h"

#if TEST_DPCPP_BACKEND_PRESENT
#include "support/utils_sycl.h"

using namespace oneapi::dpl::execution;
#endif // TEST_DPCPP_BACKEND_PRESENT
using namespace TestUtils;

// This macro may be used to analyze source data and test results in test_exclusive_scan_by_segment
// WARNING: in the case of using this macro debug output is very large.
// #define DUMP_CHECK_RESULTS

DEFINE_TEST_2(test_exclusive_scan_by_segment, BinaryPredicate, BinaryOperation)
{
    DEFINE_TEST_CONSTRUCTOR(test_exclusive_scan_by_segment, 1.0f, 1.0f)

    template <typename Iterator1, typename Iterator2, typename Iterator3, typename Size>
    void initialize_data(Iterator1 host_keys, Iterator2 host_vals, Iterator3 host_val_res, Size n)
    {
        //T keys[n] = { 1, 2, 3, 4, 1, 1, 2, 2, 3, 3, 4, 4, 1, 1, 1, ...};
        //T vals[n] = { n random numbers between 0 and 4 };

        ::std::srand(42);
        Size segment_length = 1;
        Size j = 0;
        for (Size i = 0; i != n; ++i)
        {
            host_keys[i] = j / segment_length + 1;
            host_vals[i] = rand() % 5;
            host_val_res[i] = 0;
            ++j;
            if (j == 4 * segment_length)
            {
                ++segment_length;
                j = 0;
            }
        }
    }

#ifdef DUMP_CHECK_RESULTS
    template <typename Iterator, typename Size>
    void display_param(const char* msg, Iterator it, Size n)
    {
        ::std::cout << msg;
        for (Size i = 0; i < n; ++i)
        {
            if (i > 0)
                ::std::cout << ", ";
            ::std::cout << it[i];
        }
        ::std::cout << ::std::endl;
    }
#endif // DUMP_CHECK_RESULTS

    template <typename Iterator1, typename Iterator2, typename Iterator3, typename Size, typename T,
              typename BinaryPredicateCheck = oneapi::dpl::__internal::__pstl_equal,
              typename BinaryOperationCheck = oneapi::dpl::__internal::__pstl_plus>
    void check_values(Iterator1 host_keys, Iterator2 host_vals, Iterator3 val_res, Size n, T init,
                      BinaryPredicateCheck pred = BinaryPredicateCheck(),
                      BinaryOperationCheck op = BinaryOperationCheck())
    {
        // https://docs.oneapi.io/versions/latest/onedpl/extension_api.html
        // keys:   [ 0, 0, 0, 1, 1, 1 ]
        // values: [ 1, 2, 3, 4, 5, 6 ]
        // result: [ 0, 0 + 1 = 1, 0 + 1 + 2 = 3, 0, 0 + 4 = 4, 0 + 4 + 5 = 9 ]

        if (n < 1)
            return;

        using value_type = std::decay_t<decltype(val_res[0])>;

        ::std::vector<value_type> expected_val_res(n);
        exclusive_scan_by_segment_serial(host_keys, host_vals, ::std::begin(expected_val_res), n,
            init, pred, op);

#ifdef DUMP_CHECK_RESULTS
        ::std::cout << "check_values(n = " << n << "), init = " << init << ":" << ::std::endl;
        display_param("         keys:   ", host_keys, n);
        display_param("         values: ", host_vals, n);
        display_param("         result: ", val_res, n);
        display_param("expected result: ", expected_val_res.data(), n);
#endif // DUMP_CHECK_RESULTS

        EXPECT_EQ_N(expected_val_res.data(), val_res, n, "Wrong effect from exclusive_scan_by_segment");
    }

#if TEST_DPCPP_BACKEND_PRESENT
    // specialization for hetero policy
    template <typename Policy, typename Iterator1, typename Iterator2, typename Iterator3, typename Size>
    ::std::enable_if_t<oneapi::dpl::__internal::__is_hetero_execution_policy_v<::std::decay_t<Policy>> &&
                       is_base_of_iterator_category_v<::std::random_access_iterator_tag, Iterator3>>
    operator()(Policy&& exec, Iterator1 keys_first, Iterator1 keys_last, Iterator2 vals_first, Iterator2 vals_last,
               Iterator3 val_res_first, Iterator3 val_res_last, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);
        TestDataTransfer<UDTKind::eVals, Size> host_vals(*this, n);
        TestDataTransfer<UDTKind::eRes, Size> host_val_res(*this, n);

        typedef typename ::std::iterator_traits<Iterator1>::value_type KeyT;
        typedef typename ::std::iterator_traits<Iterator2>::value_type ValT;

        const ValT zero = 0;
        const ValT init = 1;

        // call algorithm with no optional arguments
        initialize_data(host_keys.get(), host_vals.get(), host_val_res.get(), n);
        update_data(host_keys, host_vals, host_val_res);

        auto new_policy = make_new_policy<new_kernel_name<Policy, 0>>(exec);
        auto res1 =
            oneapi::dpl::exclusive_scan_by_segment(new_policy, keys_first, keys_last, vals_first, val_res_first);
        exec.queue().wait_and_throw();

        EXPECT_TRUE(std::distance(val_res_first, res1) == n, "wrong return value, device policy");
        retrieve_data(host_vals, host_val_res);
        check_values(host_keys.get(), host_vals.get(), host_val_res.get(), n, zero);

        // call algorithm with init
        initialize_data(host_keys.get(), host_vals.get(), host_val_res.get(), n);
        update_data(host_keys, host_vals, host_val_res);

        auto new_policy2 = make_new_policy<new_kernel_name<Policy, 1>>(exec);
        auto res2 =
            oneapi::dpl::exclusive_scan_by_segment(new_policy2, keys_first, keys_last, vals_first, val_res_first, init);
        exec.queue().wait_and_throw();

        EXPECT_TRUE(std::distance(val_res_first, res2) == n, "wrong return value, init, device policy");
        retrieve_data(host_vals, host_val_res);
        check_values(host_keys.get(), host_vals.get(), host_val_res.get(), n, init);

        // call algorithm with init and predicate
        initialize_data(host_keys.get(), host_vals.get(), host_val_res.get(), n);
        update_data(host_keys, host_vals, host_val_res);

        auto binary_op = [](ValT first, ValT second) { return first + second; };
        auto new_policy3 = make_new_policy<new_kernel_name<Policy, 2>>(exec);
        auto res3 = oneapi::dpl::exclusive_scan_by_segment(new_policy3, keys_first, keys_last, vals_first,
                                                           val_res_first, init, BinaryPredicate());
        exec.queue().wait_and_throw();

        EXPECT_TRUE(std::distance(val_res_first, res3) == n, "wrong return value, init and predicate, device policy");
        retrieve_data(host_vals, host_val_res);
        check_values(host_keys.get(), host_vals.get(), host_val_res.get(), n, init, BinaryPredicate());

        // call algorithm with init, predicate, and operator
        initialize_data(host_keys.get(), host_vals.get(), host_val_res.get(), n);
        update_data(host_keys, host_vals, host_val_res);

        auto new_policy4 = make_new_policy<new_kernel_name<Policy, 3>>(exec);
        auto res4 = oneapi::dpl::exclusive_scan_by_segment(new_policy4, keys_first, keys_last, vals_first,
                                                           val_res_first, init, BinaryPredicate(), BinaryOperation());
        exec.queue().wait_and_throw();

        EXPECT_TRUE(std::distance(val_res_first, res4) == n, "wrong return value, init and predicate and operator, device policy");
        retrieve_data(host_vals, host_val_res);
        check_values(host_keys.get(), host_vals.get(), host_val_res.get(), n, init, BinaryPredicate(),
                     BinaryOperation());
    }
#endif // TEST_DPCPP_BACKEND_PRESENT

    // specialization for host execution policies
    template <typename Policy, typename Iterator1, typename Iterator2, typename Iterator3, typename Size>
    ::std::enable_if_t<
#if TEST_DPCPP_BACKEND_PRESENT
        !oneapi::dpl::__internal::__is_hetero_execution_policy_v<::std::decay_t<Policy>> &&
#endif // TEST_DPCPP_BACKEND_PRESENT
            is_base_of_iterator_category_v<::std::random_access_iterator_tag, Iterator3>>
    operator()(Policy&& exec, Iterator1 keys_first, Iterator1 keys_last, Iterator2 vals_first, Iterator2 vals_last,
               Iterator3 val_res_first, Iterator3 val_res_last, Size n)
    {
        typedef typename ::std::iterator_traits<Iterator2>::value_type ValT;

        const ValT zero = 0;
        const ValT init = 1;

        // call algorithm with no optional arguments
        initialize_data(keys_first, vals_first, val_res_first, n);
        auto res1 = oneapi::dpl::exclusive_scan_by_segment(exec, keys_first, keys_last, vals_first, val_res_first);
        EXPECT_TRUE(std::distance(val_res_first, res1) == n, "wrong return value, host policy");
        check_values(keys_first, vals_first, val_res_first, n, zero);

        // call algorithm with init
        initialize_data(keys_first, vals_first, val_res_first, n);
        auto res2 =
            oneapi::dpl::exclusive_scan_by_segment(exec, keys_first, keys_last, vals_first, val_res_first, init);
        EXPECT_TRUE(std::distance(val_res_first, res2) == n, "wrong return value, init, host policy");
        check_values(keys_first, vals_first, val_res_first, n, init);

        // call algorithm with init and predicate
        initialize_data(keys_first, vals_first, val_res_first, n);
        auto res3 = oneapi::dpl::exclusive_scan_by_segment(exec, keys_first, keys_last, vals_first, val_res_first, init,
                                                           BinaryPredicate());
        EXPECT_TRUE(std::distance(val_res_first, res3) == n, "wrong return value, init and predicate, host policy");
        check_values(keys_first, vals_first, val_res_first, n, init, BinaryPredicate());

        // call algorithm with init, predicate, and operator
        initialize_data(keys_first, vals_first, val_res_first, n);
        auto res4 = oneapi::dpl::exclusive_scan_by_segment(exec, keys_first, keys_last, vals_first, val_res_first, init,
                                                           BinaryPredicate(), BinaryOperation());
        EXPECT_TRUE(std::distance(val_res_first, res4) == n, "wrong return value, init and predicate and operator, host policy");
        check_values(keys_first, vals_first, val_res_first, n, init, BinaryPredicate(), BinaryOperation());
    }

    // specialization for non-random_access iterators
    template <typename Policy, typename Iterator1, typename Iterator2, typename Iterator3, typename Size>
    ::std::enable_if_t<!is_base_of_iterator_category_v<::std::random_access_iterator_tag, Iterator3>>
    operator()(Policy&& exec, Iterator1 keys_first, Iterator1 keys_last, Iterator2 vals_first, Iterator2 vals_last,
               Iterator3 val_res_first, Iterator3 val_res_last, Size n)
    {
    }
};

int
main()
{
    {
        using ValueType = ::std::uint64_t;
        using BinaryPredicate = UserBinaryPredicate<ValueType>;
        using BinaryOperation = MaxFunctor<ValueType>;

#if TEST_DPCPP_BACKEND_PRESENT
        // Run tests for USM shared memory
        test3buffers<sycl::usm::alloc::shared,
                     test_exclusive_scan_by_segment<ValueType, BinaryPredicate, BinaryOperation>>();
        // Run tests for USM device memory
        test3buffers<sycl::usm::alloc::device,
                     test_exclusive_scan_by_segment<ValueType, BinaryPredicate, BinaryOperation>>();
#endif // TEST_DPCPP_BACKEND_PRESENT

#if TEST_DPCPP_BACKEND_PRESENT
        test_algo_three_sequences<test_exclusive_scan_by_segment<ValueType, BinaryPredicate, BinaryOperation>>();
#else
        test_algo_three_sequences<ValueType, test_exclusive_scan_by_segment<BinaryPredicate, BinaryOperation>>();
#endif // TEST_DPCPP_BACKEND_PRESENT
    }

    {
        using ValueType = MatrixPoint<float>;
        using BinaryPredicate = UserBinaryPredicate<ValueType>;
        using BinaryOperation = MaxFunctor<ValueType>;

#if TEST_DPCPP_BACKEND_PRESENT
        // Run tests for USM shared memory
        test3buffers<sycl::usm::alloc::shared,
                     test_exclusive_scan_by_segment<ValueType, BinaryPredicate, BinaryOperation>>();
        // Run tests for USM device memory
        test3buffers<sycl::usm::alloc::device,
                     test_exclusive_scan_by_segment<ValueType, BinaryPredicate, BinaryOperation>>();
#endif // TEST_DPCPP_BACKEND_PRESENT

#if TEST_DPCPP_BACKEND_PRESENT
        test_algo_three_sequences<test_exclusive_scan_by_segment<ValueType, BinaryPredicate, BinaryOperation>>();
#else
        test_algo_three_sequences<ValueType, test_exclusive_scan_by_segment<BinaryPredicate, BinaryOperation>>();
#endif // TEST_DPCPP_BACKEND_PRESENT
    }

    return TestUtils::done();
}
