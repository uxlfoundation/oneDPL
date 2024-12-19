// -*- C++ -*-
//===-- reduce_by_segment.pass.cpp ----------------------------------------===//
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

#if defined(ONEDPL_WORKAROUND_FOR_IGPU_64BIT_REDUCTION)
#undef ONEDPL_WORKAROUND_FOR_IGPU_64BIT_REDUCTION
#endif

#if defined(_ONEDPL_TEST_FORCE_WORKAROUND_FOR_IGPU_64BIT_REDUCTION)
#    define ONEDPL_WORKAROUND_FOR_IGPU_64BIT_REDUCTION _ONEDPL_TEST_FORCE_WORKAROUND_FOR_IGPU_64BIT_REDUCTION
#endif

#include "support/test_config.h"

#include "oneapi/dpl/execution"
#include "oneapi/dpl/algorithm"
#include "oneapi/dpl/numeric"
#include "oneapi/dpl/iterator"

#include "support/utils.h"
#include "support/utils_invoke.h"
#include "support/reduce_serial_impl.h"

#include <iostream>
#include <iomanip>

#if TEST_DPCPP_BACKEND_PRESENT
#    include "support/utils_sycl.h"
#    include "support/sycl_alloc_utils.h"
#endif // TEST_DPCPP_BACKEND_PRESENT
using namespace TestUtils;

// This macro may be used to analyze source data and test results in test_reduce_by_segment
// WARNING: in the case of using this macro debug output is very large.
// #define DUMP_CHECK_RESULTS

DEFINE_TEST_2(test_reduce_by_segment, BinaryPredicate, BinaryOperation)
{
    DEFINE_TEST_CONSTRUCTOR(test_reduce_by_segment, 1.0f, 1.0f)

    template <typename Iterator1, typename Iterator2, typename Iterator3, typename Iterator4, typename Size>
    void initialize_data(Iterator1 host_keys, Iterator2 host_vals, Iterator3 host_key_res, Iterator4 host_val_res,
                         Size n)
    {
        // T keys[n] = { 1, 2, 3, 4, 1, 1, 2, 2, 3, 3, 4, 4, 1, 1, 1, ...};
        // T vals[n] = { n random numbers between 0 and 4 };

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

    template <typename Iterator1, typename Iterator2, typename Iterator3, typename Iterator4, typename InputSize,
              typename OutputKeySize, typename OutputValSize,
              typename BinaryPredicateCheck = oneapi::dpl::__internal::__pstl_equal,
              typename BinaryOperationCheck = oneapi::dpl::__internal::__pstl_plus>
    void check_values(Iterator1 host_keys, Iterator2 host_vals, Iterator3 key_res, Iterator4 val_res, InputSize n,
                      OutputKeySize num_keys, OutputValSize num_vals,
                      BinaryPredicateCheck pred = BinaryPredicateCheck(),
                      BinaryOperationCheck op = BinaryOperationCheck())
    {
        // https://docs.oneapi.io/versions/latest/onedpl/extension_api.html
        // keys:   [ 0, 0, 0, 1, 1, 1 ]
        // values: [ 1, 2, 3, 4, 5, 6 ]
        // result keys: [ 0, 1 ]
        // result values: [ 1 + 2 + 3 = 6, 4 + 5 + 6 = 15 ]

        if (n < 1)
            return;

        typedef typename ::std::iterator_traits<Iterator1>::value_type KeyT;
        typedef typename ::std::iterator_traits<Iterator2>::value_type ValT;

        ::std::vector<KeyT> expected_key_res(n);
        ::std::vector<ValT> expected_val_res(n);

        ::std::size_t num_segments =
            reduce_by_segment_serial(host_keys, host_vals, ::std::begin(expected_key_res),
                ::std::begin(expected_val_res), n, pred, op);

#ifdef DUMP_CHECK_RESULTS
        ::std::cout << "check_values(n = " << n << ", segments = " << num_keys << ") : " << ::std::endl;
        display_param("           keys: ", host_keys, n);
        display_param("         values: ", host_vals, n);
        display_param("    result keys: ", key_res, num_keys);
        display_param("  result values: ", val_res, num_vals);
        display_param("  expected keys: ", expected_key_res.data(), num_segments);
        display_param("expected values: ", expected_val_res.data(), num_segments);
#endif // DUMP_CHECK_RESULTS

        EXPECT_EQ(num_segments, num_keys, "wrong key size from reduce_by_segment");
        EXPECT_EQ(num_segments, num_vals, "wrong val size from reduce_by_segment");
        EXPECT_EQ_N(expected_key_res.data(), key_res, num_keys, "incorrect keys from reduce_by_segment");
        EXPECT_EQ_N(expected_val_res.data(), val_res, num_vals, "incorrect vals from reduce_by_segment");
    }

#if TEST_DPCPP_BACKEND_PRESENT
    // specialization for hetero policy
    template <typename Policy, typename Iterator1, typename Iterator2, typename Iterator3, typename Iterator4,
              typename Size>
    ::std::enable_if_t<oneapi::dpl::__internal::__is_hetero_execution_policy_v<::std::decay_t<Policy>> &&
                       is_base_of_iterator_category_v<::std::random_access_iterator_tag, Iterator3> &&
                       is_base_of_iterator_category_v<::std::random_access_iterator_tag, Iterator4>>
    operator()(Policy&& exec, Iterator1 keys_first, Iterator1 keys_last, Iterator2 vals_first, Iterator2 vals_last,
               Iterator3 key_res_first, Iterator3 key_res_last, Iterator4 val_res_first, Iterator4 val_res_last, Size n)
    {
        TestDataTransfer<UDTKind::eKeys, Size> host_keys(*this, n);
        TestDataTransfer<UDTKind::eVals, Size> host_vals(*this, n);
        TestDataTransfer<UDTKind::eRes, Size> host_res_keys(*this, n);
        TestDataTransfer<UDTKind::eRes2, Size> host_res(*this, n);

        typedef typename ::std::iterator_traits<Iterator1>::value_type KeyT;
        typedef typename ::std::iterator_traits<Iterator2>::value_type ValT;

        initialize_data(host_keys.get(), host_vals.get(), host_res_keys.get(), host_res.get(), n);
        update_data(host_keys, host_vals, host_res_keys, host_res);

        std::pair<Iterator3, Iterator4> res;
        if constexpr (std::is_same_v<std::equal_to<KeyT>, std::decay_t<BinaryPredicate>> &&
                      std::is_same_v<std::plus<ValT>, std::decay_t<BinaryOperation>>)
        {
            res = oneapi::dpl::reduce_by_segment(exec, keys_first, keys_last, vals_first, key_res_first, val_res_first);
        }
        else if constexpr (std::is_same_v<std::plus<ValT>, std::decay_t<BinaryOperation>>)
        {
            res = oneapi::dpl::reduce_by_segment(exec, keys_first, keys_last, vals_first, key_res_first, val_res_first,
                                                 BinaryPredicate());
        }
        else
        {
            res = oneapi::dpl::reduce_by_segment(exec, keys_first, keys_last, vals_first, key_res_first, val_res_first,
                                                 BinaryPredicate(), BinaryOperation());
        }
        exec.queue().wait_and_throw();

        retrieve_data(host_keys, host_vals, host_res_keys, host_res);
        size_t segments_key_ret = ::std::distance(key_res_first, res.first);
        size_t segments_val_ret = ::std::distance(val_res_first, res.second);
        check_values(host_keys.get(), host_vals.get(), host_res_keys.get(), host_res.get(), n, segments_key_ret,
                     segments_val_ret, BinaryPredicate(), BinaryOperation());
    }
#endif

    // specialization for host execution policies
    template <typename Policy, typename Iterator1, typename Iterator2, typename Iterator3, typename Iterator4,
              typename Size>
    ::std::enable_if_t<
#if TEST_DPCPP_BACKEND_PRESENT
        !oneapi::dpl::__internal::__is_hetero_execution_policy_v<::std::decay_t<Policy>> &&
#endif
            is_base_of_iterator_category_v<::std::random_access_iterator_tag, Iterator3> &&
            is_base_of_iterator_category_v<::std::random_access_iterator_tag, Iterator4>>
    operator()(Policy&& exec, Iterator1 keys_first, Iterator1 keys_last, Iterator2 vals_first, Iterator2 vals_last,
               Iterator3 key_res_first, Iterator3 key_res_last, Iterator4 val_res_first, Iterator4 val_res_last, Size n)
    {
        typedef typename ::std::iterator_traits<Iterator1>::value_type KeyT;
        typedef typename ::std::iterator_traits<Iterator2>::value_type ValT;

        initialize_data(keys_first, vals_first, key_res_first, val_res_first, n);

        std::pair<Iterator3, Iterator4> res;
        if constexpr (std::is_same_v<std::equal_to<KeyT>, std::decay_t<BinaryPredicate>> &&
                      std::is_same_v<std::plus<ValT>, std::decay_t<BinaryOperation>>)
        {
            res = oneapi::dpl::reduce_by_segment(std::forward<Policy>(exec), keys_first, keys_last, vals_first,
                                                 key_res_first, val_res_first);
        }
        else if constexpr (std::is_same_v<std::plus<ValT>, std::decay_t<BinaryOperation>>)
        {
            res = oneapi::dpl::reduce_by_segment(std::forward<Policy>(exec), keys_first, keys_last, vals_first,
                                                 key_res_first, val_res_first, BinaryPredicate());
        }
        else
        {
            res = oneapi::dpl::reduce_by_segment(std::forward<Policy>(exec), keys_first, keys_last, vals_first,
                                                 key_res_first, val_res_first, BinaryPredicate(), BinaryOperation());
        }
        size_t segments_key_ret = ::std::distance(key_res_first, res.first);
        size_t segments_val_ret = ::std::distance(val_res_first, res.second);
        check_values(keys_first, vals_first, key_res_first, val_res_first, n, segments_key_ret, segments_val_ret,
                     BinaryPredicate(), BinaryOperation());
    }

    // specialization for non-random_access iterators
    template <typename Policy, typename Iterator1, typename Iterator2, typename Iterator3, typename Iterator4,
              typename Size>
    ::std::enable_if_t<!is_base_of_iterator_category_v<::std::random_access_iterator_tag, Iterator3> ||
                           !is_base_of_iterator_category_v<::std::random_access_iterator_tag, Iterator4>>
    operator()(Policy&& exec, Iterator1 keys_first, Iterator1 keys_last, Iterator2 vals_first, Iterator2 vals_last,
               Iterator3 key_res_first, Iterator3 key_res_last, Iterator4 val_res_first, Iterator4 val_res_last, Size n)
    {
    }
};

#if TEST_DPCPP_BACKEND_PRESENT
template <sycl::usm::alloc alloc_type, typename KernelName, typename T>
void
test_flag_pred()
{
    sycl::queue q = TestUtils::get_test_queue();

    // Initialize data
    //T keys[n1] = { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0 };
    //T vals[n1] = { 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2 };

    // keys_result = {1, 1, 1};
    // vals_result = {11, 12, 10};

    auto prepare_data = [](int n, T* key_head, T* val_head, T* key_res_head, T* val_res_head)
        {
            for (int i = 0; i < n; ++i)
            {
                key_head[i] = i % 5 == 0 ? 1 : 0;
                val_head[i] = i % 4 + 1;
            }
        };

    constexpr int n = 14;
    T key_head_on_host[n] = {};
    T val_head_on_host[n] = {};
    T key_res_head_on_host[n] = {};
    T val_res_head_on_host[n] = {};

    prepare_data(n, key_head_on_host, val_head_on_host, key_res_head_on_host, val_res_head_on_host);
    auto flag_pred = [](const auto& a, const auto& b) {
        using KeyT = ::std::decay_t<decltype(b)>;
        return b != KeyT(1);
    };
    // allocate USM memory and copying data to USM shared/device memory
    TestUtils::usm_data_transfer<alloc_type, T> dt_helper1(q, std::begin(key_head_on_host),     std::end(key_head_on_host));
    TestUtils::usm_data_transfer<alloc_type, T> dt_helper2(q, std::begin(val_head_on_host),     std::end(val_head_on_host));
    TestUtils::usm_data_transfer<alloc_type, T> dt_helper3(q, std::begin(key_res_head_on_host), std::end(key_res_head_on_host));
    TestUtils::usm_data_transfer<alloc_type, T> dt_helper4(q, std::begin(val_res_head_on_host), std::end(val_res_head_on_host));
    auto key_head     = dt_helper1.get_data();
    auto val_head     = dt_helper2.get_data();
    auto key_res_head = dt_helper3.get_data();
    auto val_res_head = dt_helper4.get_data();

    // call algorithm
    auto new_policy = oneapi::dpl::execution::make_device_policy<TestUtils::unique_kernel_name<
        TestUtils::unique_kernel_name<KernelName, 1>, TestUtils::uniq_kernel_index<alloc_type>()>>(q);
    auto res1 =
        oneapi::dpl::reduce_by_segment(new_policy, key_head, key_head + n, val_head, key_res_head, val_res_head, flag_pred, std::plus<T>());

    //retrieve result on the host and check the result
    dt_helper3.retrieve_data(key_res_head_on_host);
    dt_helper4.retrieve_data(val_res_head_on_host);

    // check values
    auto count = std::distance(key_res_head, res1.first);
    std::int64_t expected_count = 3;
    EXPECT_EQ(count, expected_count, "reduce_by_segment: incorrect number of segments");
    T expected_key(1);
    T expected_value(11);
    EXPECT_EQ(key_res_head_on_host[0], expected_key, "reduce_by_segment: wrong key");
    EXPECT_EQ(val_res_head_on_host[0], expected_value, "reduce_by_segment: wrong value");
    EXPECT_EQ(key_res_head_on_host[1], expected_key, "reduce_by_segment: wrong key");
    expected_value = T(12);
    EXPECT_EQ(val_res_head_on_host[1], expected_value, "reduce_by_segment: wrong value");
    EXPECT_EQ(key_res_head_on_host[2], expected_key, "reduce_by_segment: wrong key");
    expected_value = T(10);
    EXPECT_EQ(val_res_head_on_host[2], expected_value, "reduce_by_segment: wrong value");
}
#endif

template <bool use_device_alloc, typename ValueType, typename BinaryPredicate, typename BinaryOperation>
void
run_test_on_device()
{
#if TEST_DPCPP_BACKEND_PRESENT
    // Skip 64-byte types testing when the algorithm is broken and there is no the workaround
#if _PSTL_ICPX_TEST_RED_BY_SEG_BROKEN_64BIT_TYPES && !ONEDPL_WORKAROUND_FOR_IGPU_64BIT_REDUCTION
    if constexpr (sizeof(ValueType) != 8)
#endif
    {
        if (TestUtils::has_type_support<ValueType>(TestUtils::get_test_queue().get_device()))
        {
            constexpr sycl::usm::alloc allocation_type = use_device_alloc ? sycl::usm::alloc::device : sycl::usm::alloc::shared;
            test4buffers<allocation_type, test_reduce_by_segment<ValueType, BinaryPredicate, BinaryOperation>>();
        }
    }
#endif // TEST_DPCPP_BACKEND_PRESENT
}

template <typename ValueType, typename BinaryPredicate, typename BinaryOperation>
void
run_test_on_host()
{
#if !_PSTL_ICC_TEST_SIMD_UDS_BROKEN && !_PSTL_ICPX_TEST_RED_BY_SEG_OPTIMIZER_CRASH
#if TEST_DPCPP_BACKEND_PRESENT
    test_algo_four_sequences<test_reduce_by_segment<ValueType, BinaryPredicate, BinaryOperation>>();
#   else
    test_algo_four_sequences<ValueType, test_reduce_by_segment<BinaryPredicate, BinaryOperation>>();
#   endif
#endif // !_PSTL_ICC_TEST_SIMD_UDS_BROKEN && !_PSTL_ICPX_TEST_RED_BY_SEG_OPTIMIZER_CRASH
}

template <bool use_device_alloc, typename ValueType, typename BinaryPredicate, typename BinaryOperation>
void
run_test()
{
    run_test_on_host<ValueType, BinaryPredicate, BinaryOperation>();
    run_test_on_device<use_device_alloc, ValueType, BinaryPredicate, BinaryOperation>();
}

int
main()
{
    // On Windows, we observe incorrect results with this test with a specific compilation order of the
    // kernels. This is being filed to the compiler team. In the meantime, we can rearrange this test
    // to resolve the issue on our side.
#if _PSTL_RED_BY_SEG_WINDOWS_COMPILE_ORDER_BROKEN
    run_test</*use_device_alloc=*/true, MatrixPoint<float>, UserBinaryPredicate<MatrixPoint<float>>, MaxFunctor<MatrixPoint<float>>>();
#endif

#if TEST_DPCPP_BACKEND_PRESENT
    // test with flag pred
    test_flag_pred<sycl::usm::alloc::device, class KernelName1, std::uint64_t>();
    test_flag_pred<sycl::usm::alloc::device, class KernelName2, MatrixPoint<float>>();
#endif // TEST_DPCPP_BACKEND_PRESENT

#if !_PSTL_RED_BY_SEG_WINDOWS_COMPILE_ORDER_BROKEN
    run_test</*use_device_alloc=*/true, MatrixPoint<float>, UserBinaryPredicate<MatrixPoint<float>>, MaxFunctor<MatrixPoint<float>>>();
#endif

    run_test</*use_device_alloc=*/false, int, ::std::equal_to<int>, ::std::plus<int>>();
    run_test</*use_device_alloc=*/true, float, ::std::equal_to<float>, ::std::plus<float>>();
    run_test</*use_device_alloc=*/false, double, ::std::equal_to<double>, ::std::plus<double>>();

    // TODO investigate possible overflow: see issue #1416
    run_test_on_device</*use_device_alloc=*/true, int, ::std::equal_to<int>, ::std::multiplies<int>>();
    run_test_on_device</*use_device_alloc=*/false, float, ::std::equal_to<float>, ::std::multiplies<float>>();
    run_test_on_device</*use_device_alloc=*/true, double, ::std::equal_to<double>, ::std::multiplies<double>>();

    return TestUtils::done();
}
