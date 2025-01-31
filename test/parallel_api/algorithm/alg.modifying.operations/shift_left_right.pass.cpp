// -*- C++ -*-
//===-- shift_left_right.pass.cpp -----------------------------------------------===//
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

#include _PSTL_TEST_HEADER(execution)
#include _PSTL_TEST_HEADER(algorithm)

#include "support/utils.h"

#include <iostream>
#include <chrono>
#include <algorithm>
#include <list>
#include <iomanip>

#if  !defined(_PSTL_TEST_SHIFT_LEFT) && !defined(_PSTL_TEST_SHIFT_RIGHT)
#define _PSTL_TEST_SHIFT_LEFT
#define _PSTL_TEST_SHIFT_RIGHT
#endif

#if TEST_DPCPP_BACKEND_PRESENT
#include "support/utils_sycl.h"
#include "support/sycl_alloc_utils.h"
#endif

#if ONEDPL_USE_OPENMP_BACKEND
#include <omp.h> // omp_get_max_threads, omp_set_num_threads
#endif

template <typename... Name>
struct USMKernelName;

template <typename... Name>
struct BufferKernelName;

struct test_shift
{
    // Additional check for std::execution::par_unseq is required because standard execution policy is
    // not a host execution policy in terms of oneDPL and the eligible overload of operator() would not be found
    // while testing PSTL offload
    template <typename Policy, typename It, typename Algo>
    std::enable_if_t<oneapi::dpl::__internal::__is_host_execution_policy<std::decay_t<Policy>>::value
#if __SYCL_PSTL_OFFLOAD__
                     || std::is_same_v<std::decay_t<Policy>, std::execution::parallel_unsequenced_policy>
#endif
                     >
    operator()(Policy&& exec, It first, typename ::std::iterator_traits<It>::difference_type m,
        It first_exp, typename ::std::iterator_traits<It>::difference_type n, Algo algo)
    {
        //run a test with host policy and host itertors
        It res = algo(::std::forward<Policy>(exec), first, ::std::next(first, m), n);
        //check result
        algo.check(res, first, m, first_exp, n);
    }

#if TEST_DPCPP_BACKEND_PRESENT

#if _PSTL_SYCL_TEST_USM
    template <sycl::usm::alloc alloc_type, typename Policy, typename It, typename Algo>
    void
    test_usm(Policy&& exec, It first, typename ::std::iterator_traits<It>::difference_type m, It first_exp,
        typename ::std::iterator_traits<It>::difference_type n, Algo algo)
    {
        using _ValueType = typename ::std::iterator_traits<It>::value_type;
        using _DiffType = typename ::std::iterator_traits<It>::difference_type;

        auto queue = exec.queue();

        // allocate USM memory and copying data to USM shared/device memory
        TestUtils::usm_data_transfer<alloc_type, _ValueType> dt_helper(queue, first, m);

        auto ptr = dt_helper.get_data();
        auto het_res =
            algo(TestUtils::make_device_policy<USMKernelName<Algo, _ValueType>>(std::forward<Policy>(exec)), ptr,
                 ptr + m, n);
        _DiffType res_idx = het_res - ptr;

        //3.2 check result
        dt_helper.retrieve_data(first);
        algo.check(first + res_idx, first, m, first_exp, n);
    };

#endif

    template <typename Policy, typename It, typename Algo>
    oneapi::dpl::__internal::__enable_if_hetero_execution_policy<Policy>
    operator()(Policy&& exec, It first, typename ::std::iterator_traits<It>::difference_type m,
        It first_exp, typename ::std::iterator_traits<It>::difference_type n, Algo algo)
    {
        using _ValueType = typename std::iterator_traits<It>::value_type;
        using _DiffType = typename std::iterator_traits<It>::difference_type;
        auto buffer_policy = TestUtils::make_device_policy<BufferKernelName<_ValueType, Algo>>(exec);
        //1.1 run a test with hetero policy and host itertors
        auto res = algo(buffer_policy, first, first + m, n);
        //1.2 check result
        algo.check(res, first, m, first_exp, n);

        //2.1 run a test with hetero policy and hetero itertors
        _DiffType res_idx(0);
        {//scope for SYCL buffer lifetime
            sycl::buffer<_ValueType> buf(first, first + m);
            buf.set_final_data(first);
            buf.set_write_back(true);

            auto het_begin = oneapi::dpl::begin(buf);

            auto het_res = algo(buffer_policy, het_begin, het_begin + m, n);
            res_idx = het_res - het_begin;
        }
        //2.2 check result
        algo.check(first + res_idx, first, m, first_exp, n);

#if _PSTL_SYCL_TEST_USM
        //3. run a test with hetero policy and USM shared/device memory pointers
        test_usm<sycl::usm::alloc::shared>(exec, first, m, first_exp, n, algo);
        test_usm<sycl::usm::alloc::device>(std::forward<Policy>(exec), first, m, first_exp, n, algo);
#endif
    }
#endif
};

struct shift_left_algo
{
    template <typename Policy, typename It>
    It operator()(Policy&& exec, It first, It last, typename ::std::iterator_traits<It>::difference_type n)
    {
        return std::shift_left(::std::forward<Policy>(exec), first, last, n);
    }

    template <typename It, typename ItExp>
    void
    check(It res, It first, typename ::std::iterator_traits<It>::difference_type m, ItExp first_exp,
        typename ::std::iterator_traits<It>::difference_type n)
    {
        //if (n > 0 && n < m), returns first + (m - n). Otherwise, if n  > 0, returns first.
        //Otherwise, returns last.
        It __last = ::std::next(first, m);
        auto res_exp = (n > 0 && n < m ? ::std::next(first, m - n) : (n > 0 ? first : __last));

        EXPECT_TRUE(res_exp == res, "wrong return value of shift_left");

        if(res != first && res != __last)
        {
            EXPECT_EQ_N(first, ::std::next(first_exp, + n), m - n, "wrong effect of shift_left");
            //restore unput data
            std::copy_n(first_exp, m, first);
        }
    }
};

struct shift_right_algo
{
    template <typename Policy, typename It>
    ::std::enable_if_t<TestUtils::is_base_of_iterator_category_v<::std::bidirectional_iterator_tag, It>, It>
    operator()(Policy&& exec, It first, It last, typename ::std::iterator_traits<It>::difference_type n)
    {
        return std::shift_right(::std::forward<Policy>(exec), first, last, n);
    }
    //skip the test for non-bidirectional iterator (forward iterator, etc)
    template <typename Policy, typename It>
    ::std::enable_if_t<!TestUtils::is_base_of_iterator_category_v<::std::bidirectional_iterator_tag, It>, It>
    operator()(Policy&& exec, It first, It last, typename ::std::iterator_traits<It>::difference_type n)
    {
        return first;
    }

    template <typename It, typename ItExp>
    ::std::enable_if_t<TestUtils::is_base_of_iterator_category_v<::std::bidirectional_iterator_tag, It>>
    check(It res, It first, typename ::std::iterator_traits<It>::difference_type m, ItExp first_exp,
        typename ::std::iterator_traits<It>::difference_type n)
    {
        //if (n > 0 && n < m), returns first + n. Otherwise, if n  > 0, returns last.
        //Otherwise, returns first.
        It __last = ::std::next(first, m);
        auto res_exp = (n > 0 && n < m ? ::std::next(first, n) : (n > 0 ? __last : first));

        EXPECT_TRUE(res_exp == res, "wrong return value of shift_right");

        if (res != first && res != __last)
        {
            EXPECT_EQ_N(::std::next(first, n), first_exp, m - n, "wrong effect of shift_right");
            //restore unput data
            std::copy_n(first_exp, m, first);
        }
    }
    //skip the check for non-bidirectional iterator (forward iterator, etc)
    template <typename It, typename ItExp>
    ::std::enable_if_t<!TestUtils::is_base_of_iterator_category_v<::std::bidirectional_iterator_tag, It>>
    check(It res, It first, typename ::std::iterator_traits<It>::difference_type m, ItExp first_exp,
        typename ::std::iterator_traits<It>::difference_type n)
    {
    }
};

template <typename T, typename Size>
void
test_shift_by_type(Size m, Size n)
{
    TestUtils::Sequence<T> orig(m, [](::std::size_t v) -> T { return T(v); }); //fill data
    TestUtils::Sequence<T> in(m, [](::std::size_t v) -> T { return T(v); }); //fill data

#ifdef _PSTL_TEST_SHIFT_LEFT
    TestUtils::invoke_on_all_policies()(test_shift(), in.begin(), m, orig.begin(), n, shift_left_algo{});
#endif
#ifdef _PSTL_TEST_SHIFT_RIGHT
    TestUtils::invoke_on_all_policies()(test_shift(), in.begin(), m, orig.begin(), n, shift_right_algo{});
#endif
}

int
main()
{
#if ONEDPL_USE_OPENMP_BACKEND
    // Due to small problem sizes in this test, runtime explodes on CPUs with large core counts due to
    // small grain sizes per thread and cross-socket traffic.
    const int max_threads = omp_get_max_threads();
    const int threads_to_use = std::min(max_threads, int(32));
    omp_set_num_threads(threads_to_use);
#endif
    using ValueType = ::std::int32_t;

    const ::std::size_t N = 100000;
    for (long m = 0; m < N; m = m < 16 ? m + 1 : long(3.1415 * m))
        for (long n = 0; n < N; n = n < 16 ? n + 1 : long(3.1415 * n))
    {
       test_shift_by_type<ValueType>(m, n);
    }
#if TEST_DPCPP_BACKEND_PRESENT
    // Test both paths of the vectorized implementation in the SYCL backend. Use shift factors that will not divide
    // into the vector size to assess edge case handling.
    const std::size_t large_n = 1000000;
    const std::size_t quarter_shift = 250111;
    const std::size_t three_quarters_shift = 750203;
    test_shift_by_type<std::uint8_t>(large_n, quarter_shift);
    test_shift_by_type<std::uint8_t>(three_quarters_shift, large_n);
    test_shift_by_type<std::uint16_t>(large_n, quarter_shift);
#endif

    return TestUtils::done();
}
