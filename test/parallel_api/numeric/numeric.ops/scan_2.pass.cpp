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

// We create this additional test for test functional of inclusive scan
// and exclusive scan for an in-place and non-in-place scan variants.

#include "support/test_config.h"

#include "oneapi/dpl/execution"
#include "oneapi/dpl/numeric"
#include "oneapi/dpl/iterator"

#include "support/utils.h"
#include "support/scan_serial_impl.h"

using namespace TestUtils;

#if TEST_DPCPP_BACKEND_PRESENT
static const char kMsgInclusiveScanNormal [] = "Wrong effect from inclusive scan (non-inplace)";
static const char kMsgInclusiveScanInplace[] = "Wrong effect from inclusive scan (inplace)";
static const char kMsgExclusiveScanNormal [] = "Wrong effect from exclusive scan (non-inplace)";
static const char kMsgExclusiveScanInplace[] = "Wrong effect from exclusive scan (inplace)";

// TODO: replace data generation with random data and update check to compare result to
// the result of a serial implementation of the algorithm
template <typename Iterator1, typename Size>
void
initialize_data(Iterator1 host_keys, Size n)
{
    const Size kStartVal = 1;
    for (Size i = 0; i != n; ++i)
        host_keys[i] = kStartVal + i;
}

struct TestingAlgoritmInclusiveScan
{
    template <typename... TArgs>
    void
    call_onedpl(TArgs&&... args)
    {
        oneapi::dpl::inclusive_scan(::std::forward<TArgs>(args)...);
    }

    template <typename... TArgs>
    void
    call_serial(TArgs&&... args)
    {
        inclusive_scan_serial(::std::forward<TArgs>(args)...);
    }

    const char*
    getMsg(bool bInplace) const
    {
        return bInplace ? kMsgInclusiveScanInplace : kMsgInclusiveScanNormal;
    }
};

template <typename BinaryOp, int InitValue>
struct TestingAlgoritmInclusiveScanExt
{
    template <typename... TArgs>
    void
    call_onedpl(TArgs&&... args)
    {
        oneapi::dpl::inclusive_scan(::std::forward<TArgs>(args)..., BinaryOp(), InitValue);
    }

    template <typename... TArgs>
    void
    call_serial(TArgs&&... args)
    {
        inclusive_scan_serial(::std::forward<TArgs>(args)..., BinaryOp(), InitValue);
    }

    const char*
    getMsg(bool bInplace) const
    {
        return bInplace ? kMsgInclusiveScanInplace : kMsgInclusiveScanNormal;
    }
};

template <int InitValue>
struct TestingAlgoritmExclusiveScan
{
    template <typename... TArgs>
    void
    call_onedpl(TArgs&&... args)
    {
        oneapi::dpl::exclusive_scan(::std::forward<TArgs>(args)..., InitValue);
    }

    template <typename... TArgs>
    void
    call_serial(TArgs&&... args)
    {
        exclusive_scan_serial(::std::forward<TArgs>(args)..., InitValue);
    }

    const char*
    getMsg(bool bInplace) const
    {
        return bInplace ? kMsgExclusiveScanInplace : kMsgExclusiveScanNormal;
    }
};

template <int InitValue, typename BinaryOp>
struct TestingAlgoritmExclusiveScanExt
{
    template <typename... TArgs>
    void
    call_onedpl(TArgs&&... args)
    {
        oneapi::dpl::exclusive_scan(::std::forward<TArgs>(args)..., InitValue, BinaryOp());
    }

    template <typename... TArgs>
    void
    call_serial(TArgs&&... args)
    {
        exclusive_scan_serial(::std::forward<TArgs>(args)..., InitValue, BinaryOp());
    }

    const char*
    getMsg(bool bInplace) const
    {
        return bInplace ? kMsgExclusiveScanInplace : kMsgExclusiveScanNormal;
    }
};


DEFINE_TEST_1(test_scan_non_inplace, TestingAlgoritm)
{
    DEFINE_TEST_CONSTRUCTOR(test_scan_non_inplace, 1.0f, 1.0f)

    // specialization for hetero policy
    template <typename Policy, typename Iterator1, typename Iterator2, typename Size>
    ::std::enable_if_t<
        oneapi::dpl::__internal::__is_hetero_execution_policy_v<::std::decay_t<Policy>> &&
            is_base_of_iterator_category_v<::std::random_access_iterator_tag, Iterator1>>
    operator()(Policy&& exec,
               Iterator1 keys_first, Iterator1 keys_last,
               Iterator2 vals_first, Iterator2 vals_last,
               Size n)
    {
        using ValT = typename ::std::iterator_traits<Iterator2>::value_type;

        TestingAlgoritm testingAlgo;

        // UDTKind::eKeys is assigned with iterators Iterator1 keys_first, Iterator1 keys_last
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        // UDTKind::eVals is assigned with iterators Iterator2 vals_first, Iterator2 vals_last
        TestDataTransfer<UDTKind::eVals, Size> host_vals(*this, n);

        // Initialize source data in the buffer [keys_first, keys_last)
        // Iterators
        //      Iterator1 keys_first, Iterator1 keys_last
        // describe uninitialized memory so we should prepare test data on the host
        initialize_data(host_keys.get(), n);

        // Copy data from the buffer [keys_first, keys_last) to a device.
        // After test data is prepared we should copy data from the host into buffers
        // described by iterators
        //      Iterator1 keys_first, Iterator1 keys_last
        update_data(host_keys);

        // Now we are ready to call the tested algorithm
        testingAlgo.call_onedpl(make_new_policy<new_kernel_name<Policy, 0>>(exec), keys_first, keys_last, vals_first);

        // After the tested algorithm finished we should check the results.
        // For that, at first we copy data from the buffer described by iterators
        //      Iterator2 vals_first, Iterator2 vals_last
        // into the host memory.
        retrieve_data(host_vals);

        // ...and check results in the host memory.
        std::vector<ValT> expected(n);
        testingAlgo.call_serial(host_keys.get(), host_keys.get() + n, expected.data());
        EXPECT_EQ_N(expected.cbegin(), host_vals.get(), n, TestingAlgoritm().getMsg(false));
    }

    // specialization for host execution policies
    template <typename Policy, typename Iterator1, typename Iterator2, typename Size>
    ::std::enable_if_t<
        !oneapi::dpl::__internal::__is_hetero_execution_policy_v<::std::decay_t<Policy>> &&
            is_base_of_iterator_category_v<::std::random_access_iterator_tag, Iterator1>>
    operator()(Policy&& exec,
               Iterator1 keys_first, Iterator1 keys_last,
               Iterator2 vals_first, Iterator2 vals_last,
               Size n)
    {
        using ValT = typename ::std::iterator_traits<Iterator2>::value_type;

        TestingAlgoritm testingAlgo;

        // Initialize source data in the buffer [keys_first, keys_last)
        initialize_data(keys_first, n);

        testingAlgo.call_onedpl(exec, keys_first, keys_last, vals_first);

        std::vector<ValT> expected(n);
        testingAlgo.call_serial(keys_first, keys_last + n, expected.data());
        EXPECT_EQ_N(expected.cbegin(), vals_first, n, TestingAlgoritm().getMsg(false));
    }

    // specialization for non-random_access iterators
    template <typename Policy, typename Iterator1, typename Iterator2, typename Size>
    ::std::enable_if_t<!is_base_of_iterator_category_v<::std::random_access_iterator_tag, Iterator1>>
    operator()(Policy&& exec,
               Iterator1 keys_first, Iterator1 keys_last,
               Iterator2 vals_first, Iterator2 vals_last,
               Size n)
    {
    }
};

DEFINE_TEST_1(test_scan_inplace, TestingAlgoritm)
{
    DEFINE_TEST_CONSTRUCTOR(test_scan_inplace, 1.0f, 1.0f)

    // specialization for host execution policies
    template <typename Policy, typename Iterator1, typename Size>
    ::std::enable_if_t<
        !oneapi::dpl::__internal::__is_hetero_execution_policy_v<::std::decay_t<Policy>> &&
            is_base_of_iterator_category_v<::std::random_access_iterator_tag, Iterator1>>
    operator()(Policy&& exec, Iterator1 keys_first, Iterator1 keys_last, Size n)
    {
        using KeyT = typename ::std::iterator_traits<Iterator1>::value_type;

        TestingAlgoritm testingAlgo;

        // Initialize source data in the buffer [keys_first, keys_last)
        initialize_data(keys_first, n);
        const std::vector<KeyT> source_host_keys_state(keys_first, keys_first + n);

        // Now we are ready to call the tested algorithm
        testingAlgo.call_onedpl(exec, keys_first, keys_last, keys_first);

        std::vector<KeyT> expected(n);
        testingAlgo.call_serial(source_host_keys_state.cbegin(), source_host_keys_state.cend(), expected.data());
        EXPECT_EQ_N(expected.cbegin(), keys_first, n, testingAlgo.getMsg(true));
    }

    // specialization for hetero policy
    template <typename Policy, typename Iterator1, typename Size>
    ::std::enable_if_t<
        oneapi::dpl::__internal::__is_hetero_execution_policy_v<::std::decay_t<Policy>> &&
            is_base_of_iterator_category_v<::std::random_access_iterator_tag, Iterator1>>
    operator()(Policy&& exec, Iterator1 keys_first, Iterator1 keys_last,
               Size n)
    {
        using KeyT = typename ::std::iterator_traits<Iterator1>::value_type;

        TestingAlgoritm testingAlgo;

        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);

        // Initialize source data in the buffer [keys_first, keys_last)
        initialize_data(host_keys.get(), n);
        const std::vector<KeyT> source_host_keys_state(host_keys.get(), host_keys.get() + n);

        // Copy data from the buffer [keys_first, keys_last) to a device.
        update_data(host_keys);

        // Now we are ready to call tested algorithm
        testingAlgo.call_onedpl(make_new_policy<new_kernel_name<Policy, 0>>(exec), keys_first, keys_last, keys_first);

        retrieve_data(host_keys);

        std::vector<KeyT> expected(n);
        testingAlgo.call_serial(source_host_keys_state.cbegin(), source_host_keys_state.cend(), expected.data());
        EXPECT_EQ_N(expected.cbegin(), host_keys.get(), n, testingAlgo.getMsg(true));
    }

    // specialization for non-random_access iterators
    template <typename Policy, typename Iterator1, typename Size>
    ::std::enable_if_t<!is_base_of_iterator_category_v<::std::random_access_iterator_tag, Iterator1>>
    operator()(Policy&& exec, Iterator1 keys_first, Iterator1 keys_last, Size n)
    {
    }
};

template <sycl::usm::alloc alloc_type, typename ValueType, typename BinaryOperation>
void
run_test()
{
    // Inclusive scan
    {
        // Non inplace
        test2buffers<alloc_type, test_scan_non_inplace<ValueType, TestingAlgoritmInclusiveScan> >();
        test2buffers<alloc_type, test_scan_non_inplace<ValueType, TestingAlgoritmInclusiveScanExt<BinaryOperation, 2 > > >();

        // Inplace
        test1buffer<alloc_type, test_scan_inplace<ValueType, TestingAlgoritmInclusiveScan>>();
        test1buffer<alloc_type, test_scan_inplace<ValueType, TestingAlgoritmInclusiveScanExt<BinaryOperation, 2 > > >();
    }

    // Exclusive scan
    {
        // Non inplace
        test2buffers<alloc_type, test_scan_non_inplace<ValueType, TestingAlgoritmExclusiveScan<2>>>();
        test2buffers<alloc_type, test_scan_non_inplace<ValueType, TestingAlgoritmExclusiveScanExt<2, BinaryOperation > > >();

        // Inplace
        test1buffer<alloc_type, test_scan_inplace<ValueType, TestingAlgoritmExclusiveScan<2>>>();
        test1buffer<alloc_type, test_scan_inplace<ValueType, TestingAlgoritmExclusiveScanExt<2, BinaryOperation>>>();
    }
}
#endif // TEST_DPCPP_BACKEND_PRESENT

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT

    using ValueType = int;
    using BinaryOperation = ::std::plus<ValueType>;

    // Run tests for USM shared memory
    run_test<sycl::usm::alloc::shared, ValueType, BinaryOperation>();
    // Run tests for USM device memory
    run_test<sycl::usm::alloc::device, ValueType, BinaryOperation>();

#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
