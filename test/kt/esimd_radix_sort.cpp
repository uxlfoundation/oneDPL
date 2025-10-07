// -*- C++ -*-
//===-- esimd_radix_sort.cpp -----------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../support/test_config.h"
<<<<<<< HEAD
#include "kt_radix_sort_test_utils.h"

#if TEST_DPCPP_BACKEND_PRESENT

#include <oneapi/dpl/experimental/kernel_templates>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#if _ENABLE_RANGES_TESTING
#    include <oneapi/dpl/ranges>
#endif
#endif // TEST_DPCPP_BACKEND_PRESENT

#include "../support/utils.h"

#if TEST_DPCPP_BACKEND_PRESENT
#if __has_include(<sycl/sycl.hpp>)
#    include <sycl/sycl.hpp>
#else
#    include <CL/sycl.hpp>
#endif

#include "../support/sycl_alloc_utils.h"

#include <vector>
#include <algorithm>
#include <string>
#include <cstdint>
=======

#include <oneapi/dpl/experimental/kernel_templates>
#if _ENABLE_RANGES_TESTING
#    include <oneapi/dpl/ranges>
#endif

#include <vector>
#include <string>
#include <cstdint>
#include <cstdlib>
#include <algorithm>
>>>>>>> origin/main

#if LOG_TEST_INFO
#include <iostream>
#endif

<<<<<<< HEAD
#ifndef TEST_DATA_TYPE
#    define TEST_DATA_TYPE int
#endif

#ifndef TEST_DATA_PER_WORK_ITEM
#    define TEST_DATA_PER_WORK_ITEM 256
#endif

#ifndef TEST_WORK_GROUP_SIZE
#    define TEST_WORK_GROUP_SIZE 64
#endif

using ParamType = oneapi::dpl::experimental::kt::kernel_param<TEST_DATA_PER_WORK_ITEM, TEST_WORK_GROUP_SIZE>;
constexpr ParamType kernel_parameters;
=======
#if __has_include(<sycl/sycl.hpp>)
#    include <sycl/sycl.hpp>
#else
#    include <CL/sycl.hpp>
#endif

#include "../support/utils.h"
#include "../support/sycl_alloc_utils.h"

#include "esimd_radix_sort_utils.h"
>>>>>>> origin/main

#if _ENABLE_RANGES_TESTING
template <typename T, bool IsAscending, std::uint8_t RadixBits, typename KernelParam>
void
test_all_view(sycl::queue q, std::size_t size, KernelParam param)
{
#if LOG_TEST_INFO
    std::cout << "\ttest_all_view(" << size << ") : " << TypeInfo().name<T>() << std::endl;
#endif
    std::vector<T> input(size);
<<<<<<< HEAD
    generate_data(input.data(), size, 42);
=======
    TestUtils::generate_arithmetic_data(input.data(), size, 42);
>>>>>>> origin/main
    std::vector<T> ref(input);
    std::stable_sort(std::begin(ref), std::end(ref), Compare<T, IsAscending>{});
    {
        sycl::buffer<T> buf(input.data(), input.size());
        oneapi::dpl::experimental::ranges::all_view<T, sycl::access::mode::read_write> view(buf);
<<<<<<< HEAD
        oneapi::dpl::experimental::kt::esimd::radix_sort<IsAscending>(q, view, param).wait();
=======
        oneapi::dpl::experimental::kt::gpu::esimd::radix_sort<IsAscending>(q, view, param).wait();
>>>>>>> origin/main
    }

    std::string msg = "wrong results with all_view, n: " + std::to_string(size);
    EXPECT_EQ_RANGES(ref, input, msg.c_str());
}

template <typename T, bool IsAscending, std::uint8_t RadixBits, typename KernelParam>
void
test_subrange_view(sycl::queue q, std::size_t size, KernelParam param)
{
#if LOG_TEST_INFO
    std::cout << "\ttest_subrange_view<T, " << IsAscending << ">(" << size << ") : " << TypeInfo().name<T>()
              << std::endl;
#endif
    std::vector<T> expected(size);
<<<<<<< HEAD
    generate_data(expected.data(), size, 42);
=======
    TestUtils::generate_arithmetic_data(expected.data(), size, 42);
>>>>>>> origin/main

    TestUtils::usm_data_transfer<sycl::usm::alloc::device, T> dt_input(q, expected.begin(), expected.end());

    std::stable_sort(expected.begin(), expected.end(), Compare<T, IsAscending>{});

    oneapi::dpl::experimental::ranges::views::subrange view(dt_input.get_data(), dt_input.get_data() + size);
<<<<<<< HEAD
    oneapi::dpl::experimental::kt::esimd::radix_sort<IsAscending>(q, view, param).wait();
=======
    oneapi::dpl::experimental::kt::gpu::esimd::radix_sort<IsAscending>(q, view, param).wait();
>>>>>>> origin/main

    std::vector<T> actual(size);
    dt_input.retrieve_data(actual.begin());

    std::string msg = "wrong results with views::subrange, n: " + std::to_string(size);
    EXPECT_EQ_N(expected.begin(), actual.begin(), size, msg.c_str());
}

#endif // _ENABLE_RANGES_TESTING

template <typename T, bool IsAscending, std::uint8_t RadixBits, sycl::usm::alloc _alloc_type, typename KernelParam>
void
test_usm(sycl::queue q, std::size_t size, KernelParam param)
{
#if LOG_TEST_INFO
    std::cout << "\t\ttest_usm<" << TypeInfo().name<T>() << ", " << USMAllocPresentation().name<_alloc_type>() << ", "
              << IsAscending << ">(" << size << ");" << std::endl;
#endif
    std::vector<T> expected(size);
<<<<<<< HEAD
    generate_data(expected.data(), size, 42);
=======
    TestUtils::generate_arithmetic_data(expected.data(), size, 42);
>>>>>>> origin/main

    TestUtils::usm_data_transfer<_alloc_type, T> dt_input(q, expected.begin(), expected.end());

    std::stable_sort(expected.begin(), expected.end(), Compare<T, IsAscending>{});

<<<<<<< HEAD
    oneapi::dpl::experimental::kt::esimd::radix_sort<IsAscending>(q, dt_input.get_data(), dt_input.get_data() + size,
=======
    oneapi::dpl::experimental::kt::gpu::esimd::radix_sort<IsAscending>(q, dt_input.get_data(), dt_input.get_data() + size,
>>>>>>> origin/main
                                                                  param)
        .wait();

    std::vector<T> actual(size);
    dt_input.retrieve_data(actual.begin());

    std::string msg = "wrong results with USM, n: " + std::to_string(size);
    EXPECT_EQ_N(expected.begin(), actual.begin(), size, msg.c_str());
}

template <typename T, bool IsAscending, std::uint8_t RadixBits, typename KernelParam>
void
test_sycl_iterators(sycl::queue q, std::size_t size, KernelParam param)
{
#if LOG_TEST_INFO
    std::cout << "\t\ttest_sycl_iterators<" << TypeInfo().name<T>() << ">(" << size << ");" << std::endl;
#endif
    std::vector<T> input(size);
<<<<<<< HEAD
    generate_data(input.data(), size, 42);
=======
    TestUtils::generate_arithmetic_data(input.data(), size, 42);
>>>>>>> origin/main
    std::vector<T> ref(input);
    std::stable_sort(std::begin(ref), std::end(ref), Compare<T, IsAscending>{});
    {
        sycl::buffer<T> buf(input.data(), input.size());
<<<<<<< HEAD
        oneapi::dpl::experimental::kt::esimd::radix_sort<IsAscending>(q, oneapi::dpl::begin(buf), oneapi::dpl::end(buf),
                                                                      param)
            .wait();
=======

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif
        // Deprecated namespace is used deliberatelly to make sure the functionality is still available
        oneapi::dpl::experimental::kt::esimd::radix_sort<IsAscending>(q, oneapi::dpl::begin(buf), oneapi::dpl::end(buf),
                                                                      param)
            .wait();
#ifdef __clang__
#pragma clang diagnostic pop
#endif
>>>>>>> origin/main
    }

    std::string msg = "wrong results with oneapi::dpl::begin/end, n: " + std::to_string(size);
    EXPECT_EQ_RANGES(ref, input, msg.c_str());
}

template <typename T, bool IsAscending, std::uint8_t RadixBits, typename KernelParam>
void
<<<<<<< HEAD
test_small_sizes(sycl::queue q, KernelParam param)
{
    std::vector<uint32_t> input = {5, 11, 0, 17, 0};
    std::vector<uint32_t> ref(input);

    oneapi::dpl::experimental::kt::esimd::radix_sort<Ascending, RadixBits>(q, oneapi::dpl::begin(input),
=======
test_sycl_buffer(sycl::queue q, std::size_t size, KernelParam param)
{
#if LOG_TEST_INFO
    std::cout << "\t\ttest_sycl_buffer<" << TypeInfo().name<T>() << ">(" << size << ");" << std::endl;
#endif
    std::vector<T> input(size);
    TestUtils::generate_arithmetic_data(input.data(), size, 42);
    std::vector<T> ref(input);
    std::stable_sort(std::begin(ref), std::end(ref), Compare<T, IsAscending>{});
    {
        sycl::buffer<T> buf(input.data(), input.size());
        oneapi::dpl::experimental::kt::gpu::esimd::radix_sort<IsAscending>(q, buf, param).wait();
    }

    std::string msg = "wrong results with sycl::buffer, n: " + std::to_string(size);
    EXPECT_EQ_RANGES(ref, input, msg.c_str());
}

template <typename T, bool IsAscending, std::uint8_t RadixBits, typename KernelParam>
void
test_small_sizes(sycl::queue q, KernelParam param)
{
    constexpr int size = 8;
    std::vector<T> input(size);
    TestUtils::generate_arithmetic_data(input.data(), size, 42);
    std::vector<T> ref(input);

    oneapi::dpl::experimental::kt::gpu::esimd::radix_sort<IsAscending, RadixBits>(q, oneapi::dpl::begin(input),
>>>>>>> origin/main
                                                                           oneapi::dpl::begin(input), param)
        .wait();
    EXPECT_EQ_RANGES(ref, input, "sort modified input data when size == 0");

<<<<<<< HEAD
    oneapi::dpl::experimental::kt::esimd::radix_sort<Ascending, RadixBits>(q, oneapi::dpl::begin(input),
=======
    oneapi::dpl::experimental::kt::gpu::esimd::radix_sort<IsAscending, RadixBits>(q, oneapi::dpl::begin(input),
>>>>>>> origin/main
                                                                           oneapi::dpl::begin(input) + 1, param)
        .wait();
    EXPECT_EQ_RANGES(ref, input, "sort modified input data when size == 1");
}

template <typename T, bool IsAscending, std::uint8_t RadixBits, typename KernelParam>
void
test_general_cases(sycl::queue q, std::size_t size, KernelParam param)
{
<<<<<<< HEAD
    test_usm<T, IsAscending, RadixBits, sycl::usm::alloc::shared>(q, size, param);
    test_usm<T, IsAscending, RadixBits, sycl::usm::alloc::device>(q, size, param);
    test_sycl_iterators<T, IsAscending, RadixBits>(q, size, param);
#if _ENABLE_RANGES_TESTING
    test_all_view<T, IsAscending, RadixBits>(q, size, param);
    test_subrange_view<T, IsAscending, RadixBits>(q, size, param);
#endif // _ENABLE_RANGES_TESTING
}
#endif // TEST_DPCPP_BACKEND_PRESENT
=======
    test_usm<T, IsAscending, RadixBits, sycl::usm::alloc::shared>(q, size, TestUtils::create_new_kernel_param_idx<0>(param));
    test_usm<T, IsAscending, RadixBits, sycl::usm::alloc::device>(q, size, TestUtils::create_new_kernel_param_idx<1>(param));
    test_sycl_iterators<T, IsAscending, RadixBits>(q, size, TestUtils::create_new_kernel_param_idx<2>(param));
    test_sycl_buffer<T, IsAscending, RadixBits>(q, size, TestUtils::create_new_kernel_param_idx<3>(param));
#if _ENABLE_RANGES_TESTING
    test_all_view<T, IsAscending, RadixBits>(q, size, TestUtils::create_new_kernel_param_idx<4>(param));
    test_subrange_view<T, IsAscending, RadixBits>(q, size, TestUtils::create_new_kernel_param_idx<5>(param));
#endif // _ENABLE_RANGES_TESTING
}
>>>>>>> origin/main

template <typename T, typename KernelParam>
bool
can_run_test(sycl::queue q, KernelParam param)
{
    const auto max_slm_size = q.get_device().template get_info<sycl::info::device::local_mem_size>();
    // skip tests with error: LLVM ERROR: SLM size exceeds target limits
    return sizeof(T) * param.data_per_workitem * param.workgroup_size < max_slm_size;
}

int
main()
{
<<<<<<< HEAD
#if TEST_DPCPP_BACKEND_PRESENT
    auto q = TestUtils::get_test_queue();
    bool run_test = can_run_test<TEST_DATA_TYPE>(q, kernel_parameters);
    if (run_test)
    {
        const std::vector<std::size_t> sizes = {
            1,       6,         16,      43,        256,           316,           2048,
            5072,    8192,      14001,   1 << 14,   (1 << 14) + 1, 50000,         67543,
            100'000, 1 << 17,   179'581, 250'000,   1 << 18,       (1 << 18) + 1, 500'000,
            888'235, 1'000'000, 1 << 20, 10'000'000};

        try
        {
#if TEST_LONG_RUN
            for (auto size : sizes)
            {
                test_general_cases<TEST_DATA_TYPE, Ascending, TestRadixBits>(q, size, kernel_parameters);
                test_general_cases<TEST_DATA_TYPE, Descending, TestRadixBits>(q, size, kernel_parameters);
            }
            test_small_sizes<TEST_DATA_TYPE, Ascending, TestRadixBits>(q, kernel_parameters);
#else
            for (auto size : sizes)
            {
                test_usm<TEST_DATA_TYPE, Ascending, TestRadixBits, sycl::usm::alloc::shared>(q, size, kernel_parameters);
                test_usm<TEST_DATA_TYPE, Descending, TestRadixBits, sycl::usm::alloc::shared>(q, size, kernel_parameters);
            }
#endif // TEST_LONG_RUN
=======
    constexpr oneapi::dpl::experimental::kt::kernel_param<TEST_DATA_PER_WORK_ITEM, TEST_WORK_GROUP_SIZE> params;
    auto q = TestUtils::get_test_queue();
    bool run_test = can_run_test<decltype(params), TEST_KEY_TYPE>(q, params);
    if (run_test)
    {
        try
        {
            for (auto size : sort_sizes)
            {
                test_general_cases<TEST_KEY_TYPE, Ascending, TestRadixBits>(
                    q, size, TestUtils::create_new_kernel_param_idx<0>(params));
                test_general_cases<TEST_KEY_TYPE, Descending, TestRadixBits>(
                    q, size, TestUtils::create_new_kernel_param_idx<1>(params));
            }
            test_small_sizes<TEST_KEY_TYPE, Ascending, TestRadixBits>(q, TestUtils::create_new_kernel_param_idx<3>(params));
>>>>>>> origin/main
        }
        catch (const ::std::exception& exc)
        {
            std::cerr << "Exception: " << exc.what() << std::endl;
            return EXIT_FAILURE;
        }
    }
<<<<<<< HEAD
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT && run_test);
=======

    return TestUtils::done(run_test);
>>>>>>> origin/main
}
