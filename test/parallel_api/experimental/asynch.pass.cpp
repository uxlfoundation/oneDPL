// -*- C++ -*-
//===-- async.pass.cpp ----------------------------------------------------===//
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

#if TEST_DPCPP_BACKEND_PRESENT
#    include "oneapi/dpl/async"
#endif // TEST_DPCPP_BACKEND_PRESENT
#include "oneapi/dpl/execution"
#include "oneapi/dpl/iterator"

#include "support/utils.h"

#include <iostream>
#include <iomanip>
#include <numeric>

#if TEST_DPCPP_BACKEND_PRESENT
#include "support/sycl_alloc_utils.h"

template <size_t idx>
class Copy;

template <size_t idx>
class Fill;

class ForEach1;

template <size_t idx>
class Transform;

template <size_t idx>
class Reduce;

template <size_t idx>
class Scan;

class Sort;

template <size_t idx>
class Async;

struct PreIncrementFO
{
    void operator()(int& e) const
    {
        ++e;
    }
};

struct DivByTwoFO
{
    auto operator()(const int& e) const
    {
        return e / 2;
    }
};

template <typename T>
struct MultiplyByAlphaFO
{
    T alpha;

    auto operator()(int e) const
    {
        return alpha * e;
    }
};

struct MultiplyByTenFO
{
    template <typename T>
    auto
    operator()(T x) const
    {
        return x * 10;
    }
};


template <typename Policy>
void test1_with_buffers(Policy&& exec)
{
    const int n = 100;

    sycl::buffer<int> x{n};
    sycl::buffer<int> y{n};
    sycl::buffer<int> z{n};

    auto res_1a = oneapi::dpl::experimental::copy_async(CLONE_TEST_POLICY_NAME(exec, Copy<1>),
                                                        oneapi::dpl::counting_iterator<int>(0),
                                                        oneapi::dpl::counting_iterator<int>(n),
                                                        oneapi::dpl::begin(x)); // x = [0..n]

    auto res_1b = oneapi::dpl::experimental::fill_async(CLONE_TEST_POLICY_NAME(exec, Fill<1>), oneapi::dpl::begin(y),
                                                        oneapi::dpl::end(y), 7); // y = [7..7]

    auto res_2a =
        oneapi::dpl::experimental::for_each_async(CLONE_TEST_POLICY_NAME(exec, ForEach1), oneapi::dpl::begin(x),
                                                  oneapi::dpl::end(x), PreIncrementFO{}, res_1a); // x = [1..n]

    auto res_2b = oneapi::dpl::experimental::transform_async(CLONE_TEST_POLICY_NAME(exec, Transform<1>),
                                                             oneapi::dpl::begin(y), oneapi::dpl::end(y),
                                                             oneapi::dpl::begin(y), DivByTwoFO{}, res_1b); // y = [3..3]

    auto res_3 = oneapi::dpl::experimental::transform_async(
        CLONE_TEST_POLICY_NAME(exec, Transform<2>), oneapi::dpl::begin(x), oneapi::dpl::end(x), oneapi::dpl::begin(y),
        oneapi::dpl::begin(z), std::plus<int>(), res_2a, res_2b); // z = [4..n+3]

    auto alpha = oneapi::dpl::experimental::reduce_async(CLONE_TEST_POLICY_NAME(exec, Reduce<1>), oneapi::dpl::begin(x),
                                                         oneapi::dpl::end(x), 0, std::plus<int>(), res_2a)
                     .get(); // alpha = n*(n+1)/2

    auto beta = oneapi::dpl::experimental::transform_reduce_async(
        CLONE_TEST_POLICY_NAME(exec, Reduce<2>), oneapi::dpl::begin(z), oneapi::dpl::end(z), 0, std::plus<int>(),
        MultiplyByAlphaFO<decltype(alpha)>{alpha});

    auto gamma = oneapi::dpl::experimental::transform_inclusive_scan_async(
        CLONE_TEST_POLICY_NAME(exec, Scan<0>), oneapi::dpl::begin(x), oneapi::dpl::end(x), oneapi::dpl::begin(y),
        std::plus<int>(), MultiplyByTenFO{}, 0);

    auto delta = oneapi::dpl::experimental::sort_async(CLONE_TEST_POLICY_NAME(exec, Sort), oneapi::dpl::begin(y),
                                                       oneapi::dpl::end(y), std::greater<int>(), gamma);

    int small_nonzero_values[3] = {2, 3, 4};
    sycl::buffer small_nonzero{small_nonzero_values, sycl::range{3}};

    auto epsilon = oneapi::dpl::experimental::reduce_async(
        CLONE_TEST_POLICY_NAME(exec, Reduce<3>), oneapi::dpl::begin(small_nonzero), oneapi::dpl::end(small_nonzero), 1,
        std::multiplies<int>()); // epsilon = 1 * 2 * 3 * 4 = 24

    oneapi::dpl::experimental::wait_for_all(sycl::event{}, beta, gamma, delta, epsilon);

    const int expected1 = (n * (n + 1) / 2) * ((n + 3) * (n + 4) / 2 - 6);
    const int expected2 = (n * (n + 1) / 2) * 10;
    auto result1 = beta.get();
    auto result2 = y.get_host_access(sycl::read_only)[0];

    EXPECT_TRUE(result1 == expected1 && result2 == expected2, "wrong effect from async test (I) with sycl buffer");

    auto actual_epsilon = epsilon.get();
    auto expected_epsilon = 1 * 2 * 3 * 4;
    EXPECT_EQ(expected_epsilon, actual_epsilon, "wrong result for reduce_async with multiply binary_op");
}

template <typename Policy>
void test2_with_buffers(Policy&& exec)
{
    const size_t n = 100;

    sycl::buffer<float> x{n};
    sycl::buffer<float> y{n};
    sycl::buffer<float> z{n};

    auto res_1a = oneapi::dpl::experimental::copy_async(
        CLONE_TEST_POLICY_NAME(exec, Copy<21>), oneapi::dpl::counting_iterator<int>(0),
        oneapi::dpl::counting_iterator<int>(n), oneapi::dpl::begin(x)); // x = [1..n]

    auto alpha = 1.0f;
    auto beta = oneapi::dpl::experimental::transform_inclusive_scan_async(
        CLONE_TEST_POLICY_NAME(exec, Scan<21>), oneapi::dpl::begin(x), oneapi::dpl::end(x), oneapi::dpl::begin(y),
        std::plus<float>(), MultiplyByAlphaFO<decltype(alpha)>{alpha}, 0.0f, res_1a);

    auto res_1b = oneapi::dpl::experimental::fill_async(CLONE_TEST_POLICY_NAME(exec, Fill<21>), oneapi::dpl::begin(x),
                                                        oneapi::dpl::end(x), -1.0f, beta);

    auto input1 = oneapi::dpl::counting_iterator<int>(0);
    auto gamma = oneapi::dpl::experimental::inclusive_scan_async(
        CLONE_TEST_POLICY_NAME(exec, Scan<22>), input1, input1 + n, oneapi::dpl::begin(z), std::plus<float>(), 0.0f);

    auto result1 = gamma.get().get_buffer().get_host_access(sycl::read_only)[n - 1];
    auto result2 = beta.get().get_buffer().get_host_access(sycl::read_only)[n - 1];

    const float expected1 = static_cast<float>(n * (n - 1) / 2);
    EXPECT_TRUE(fabs(result1 - expected1) <= 0.001f && fabs(result2 - expected1) <= 0.001f,
                "wrong effect from async test (II) with sycl buffer");
}

// TODO: Extend tests by checking true async behavior in more detail
template <sycl::usm::alloc alloc_type, typename Policy>
void
test_with_usm(Policy&& exec)
{
    constexpr int n = 1024;
    constexpr int n_small = 13;

    // Initialize data
    auto prepare_data = [](int n, std::uint64_t* data1, std::uint64_t* data2) {
        for (int i = 0; i != n - 1; ++i)
        {
            data1[i] = i % 4 + 1;
            data2[i] = data1[i] + 1;
            if (i > 3 && i != n - 2)
            {
                ++i;
                data1[i] = data1[i - 1];
                data2[i] = data2[i - 1];
            }
        }
        data1[n - 1] = 0;
        data2[n - 1] = 0;
    };

    std::uint64_t data1_on_host[n] = {};
    std::uint64_t data2_on_host[n] = {};
    prepare_data(n, data1_on_host, data2_on_host);

    // allocate USM memory and copying data to USM shared/device memory
    TestUtils::usm_data_transfer<alloc_type, std::uint64_t> dt_helper1(exec, std::begin(data1_on_host),
                                                                       std::end(data1_on_host));
    TestUtils::usm_data_transfer<alloc_type, std::uint64_t> dt_helper2(exec, std::begin(data2_on_host),
                                                                       std::end(data2_on_host));
    auto data1 = dt_helper1.get_data();
    auto data2 = dt_helper2.get_data();

    // compute reference values
    const std::uint64_t ref1 = std::inner_product(data2_on_host, data2_on_host + n, data1_on_host, 0);
    const std::uint64_t ref2 = std::accumulate(data1_on_host, data1_on_host + n_small, 0);

    // call first algorithm
    using _NewKernelName1 = TestUtils::unique_kernel_name<Async<1>, TestUtils::uniq_kernel_index<alloc_type>()>;
    auto fut1 = oneapi::dpl::experimental::transform_reduce_async(CLONE_TEST_POLICY_NAME(exec, _NewKernelName1), data2,
                                                                  data2 + n, data1, 0, std::plus<std::uint64_t>(),
                                                                  std::multiplies<std::uint64_t>());

    // call second algorithm and wait for result
    using _NewKernelName2 = TestUtils::unique_kernel_name<Async<2>, TestUtils::uniq_kernel_index<alloc_type>()>;
    auto res2 =
        oneapi::dpl::experimental::reduce_async(CLONE_TEST_POLICY_NAME(exec, _NewKernelName2), data1, data1 + n_small)
            .get();

    // call third algorithm that has to wait for first to complete
    using _NewKernelName3 = TestUtils::unique_kernel_name<Async<3>, TestUtils::uniq_kernel_index<alloc_type>()>;
    auto sort_async_result =
        oneapi::dpl::experimental::sort_async(CLONE_TEST_POLICY_NAME(exec, _NewKernelName3), data2, data2 + n, fut1);
    sort_async_result.wait();

    // check values
    auto res1 = fut1.get();
    EXPECT_TRUE(res1 == ref1, "wrong effect from async transform reduce with usm");
    EXPECT_TRUE(res2 == ref2, "wrong effect from async reduce with usm");
}

template <typename Policy>
void test_impl(Policy&& exec)
{
    try
    {
        test1_with_buffers(CLONE_TEST_POLICY(exec));
        test2_with_buffers(CLONE_TEST_POLICY(exec));

        // Run tests for USM shared/device memory
        test_with_usm<sycl::usm::alloc::shared>(CLONE_TEST_POLICY(exec));
        test_with_usm<sycl::usm::alloc::device>(CLONE_TEST_POLICY(exec));
    }
    catch (const std::exception& exc)
    {
        std::cerr << "Exception: " << exc.what() << std::endl;
        return EXIT_FAILURE;
    }
}
#endif // #if TEST_DPCPP_BACKEND_PRESENT

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT

    auto policy = TestUtils::get_dpcpp_test_policy();
    test_impl(policy);

#if TEST_CHECK_COMPILATION_WITH_DIFF_POLICY_VAL_CATEGORY
    TestUtils::check_compilation(policy, [](auto&& policy) { test_impl(std::forward<decltype(policy)>(policy)); });
#endif
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
