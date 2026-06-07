// -*- C++ -*-
//===-- radix_sort_by_key.cpp ---------------------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../support/test_config.h"

#include <oneapi/dpl/experimental/kernel_templates>
#include <oneapi/dpl/iterator>

#include <vector>
#include <string>
#include <cstdlib>

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

#include "../support/utils.h"
#include "../support/sycl_alloc_utils.h"

#include "radix_sort_utils.h"

template<typename KeyT, typename ValueT, bool isAscending, std::uint32_t RadixBits, typename KernelParam>
void test_sycl_buffer(sycl::queue q, std::size_t size, KernelParam param)
{
    std::vector<KeyT> expected_keys(size);
    std::vector<ValueT> expected_values(size);
    TestUtils::generate_arithmetic_data(expected_keys.data(), size, 6);
    TestUtils::generate_arithmetic_data(expected_values.data(), size, 7);

    std::vector<KeyT> actual_keys(expected_keys);
    std::vector<ValueT> actual_values(expected_values);
    {
        sycl::buffer<KeyT> keys(actual_keys.data(), actual_keys.size());
        sycl::buffer<ValueT> values(actual_values.data(), actual_values.size());
        kt_ns::radix_sort_by_key<isAscending, RadixBits>(q, keys, values, param).wait();
    }

    auto expected_first  = oneapi::dpl::make_zip_iterator(std::begin(expected_keys), std::begin(expected_values));
    std::stable_sort(expected_first, expected_first + size, CompareKey<isAscending>{});

    std::string parameters_msg = ", n: " + std::to_string(size) +
                                 ", sizeof(key): " + std::to_string(sizeof(KeyT)) +
                                 ", sizeof(value): " + std::to_string(sizeof(ValueT)) +
                                 ", isAscending: " +  std::to_string(isAscending) +
                                 ", RadixBits: " + std::to_string(RadixBits);
    std::string msg = "wrong results with sycl::buffer (keys)" + parameters_msg;
    EXPECT_EQ_N(expected_keys.begin(), actual_keys.begin(), size, msg.c_str());
    msg = "wrong results with sycl::buffer (values)" + parameters_msg;
    EXPECT_EQ_N(expected_values.begin(), actual_values.begin(), size, msg.c_str());
}

template<typename KeyT, typename ValueT, bool isAscending, std::uint32_t RadixBits, sycl::usm::alloc _alloc_type, typename KernelParam>
void test_usm(sycl::queue q, std::size_t size, KernelParam param)
{
    std::vector<KeyT> expected_keys(size);
    std::vector<ValueT> expected_values(size);
    TestUtils::generate_arithmetic_data(expected_keys.data(), size, 6);
    TestUtils::generate_arithmetic_data(expected_values.data(), size, 7);

    TestUtils::usm_data_transfer<_alloc_type, KeyT> keys(q, expected_keys.begin(), expected_keys.end());
    TestUtils::usm_data_transfer<_alloc_type, ValueT> values(q, expected_values.begin(), expected_values.end());

    auto expected_first  = oneapi::dpl::make_zip_iterator(std::begin(expected_keys), std::begin(expected_values));
    std::stable_sort(expected_first, expected_first + size, CompareKey<isAscending>{});

    kt_ns::radix_sort_by_key<isAscending, RadixBits>(q, keys.get_data(), keys.get_data() + size, values.get_data(),
                                                     param)
        .wait();

    std::vector<KeyT> actual_keys(size);
    std::vector<ValueT> actual_values(size);
    keys.retrieve_data(actual_keys.begin());
    values.retrieve_data(actual_values.begin());

    std::string parameters_msg = ", n: " + std::to_string(size) +
                                 ", sizeof(key): " + std::to_string(sizeof(KeyT)) +
                                 ", sizeof(value): " + std::to_string(sizeof(ValueT)) +
                                 ", isAscending: " +  std::to_string(isAscending) +
                                 ", RadixBits: " + std::to_string(RadixBits);
    std::string msg = "wrong results with USM (keys)" + parameters_msg;
    EXPECT_EQ_N(expected_keys.begin(), actual_keys.begin(), size, msg.c_str());
    msg = "wrong results with USM (values)" + parameters_msg;
    EXPECT_EQ_N(expected_values.begin(), actual_values.begin(), size, msg.c_str());
}

// Test that -0 and +0 keys are treated as equal.
template <typename KeyT, typename ValueT, bool isAscending, std::uint32_t RadixBits, typename KernelParam>
void
test_negative_zero_stability(sycl::queue q, std::size_t size, KernelParam param)
{
    std::vector<KeyT> expected_keys(size);
    std::vector<ValueT> expected_values(size);
    for (std::size_t i = 0; i < size; ++i)
    {
        expected_keys[i] = (i % 2 == 0) ? -KeyT(0.0) : KeyT(0.0);
        expected_values[i] = static_cast<ValueT>(i);
    }

    TestUtils::usm_data_transfer<sycl::usm::alloc::device, KeyT> dt_keys(q, expected_keys.begin(), expected_keys.end());
    TestUtils::usm_data_transfer<sycl::usm::alloc::device, ValueT> dt_values(q, expected_values.begin(),
                                                                             expected_values.end());
    kt_ns::radix_sort_by_key<isAscending, RadixBits>(q, dt_keys.get_data(), dt_keys.get_data() + size,
                                                     dt_values.get_data(), param)
        .wait();

    std::vector<KeyT> actual_keys(size);
    std::vector<ValueT> actual_values(size);
    dt_keys.retrieve_data(actual_keys.begin());
    dt_values.retrieve_data(actual_values.begin());

    std::string msg = "negative zero stability broken, n: " + std::to_string(size);
    EXPECT_EQ_N(expected_keys.begin(), actual_keys.begin(), size, msg.c_str());
    EXPECT_EQ_N(expected_values.begin(), actual_values.begin(), size, msg.c_str());
}

// Test with constrained-range keys to exercise single-bin direct copy optimization.
template <typename KeyT, typename ValueT, bool IsAscending, std::uint8_t RadixBits, typename KernelParam>
void
test_constrained_range_by_key(sycl::queue q, std::size_t size, KernelParam param)
{
    std::vector<KeyT> expected_keys(size);
    std::vector<ValueT> expected_values(size);
    std::default_random_engine gen{73};
    std::uniform_int_distribution<std::uint32_t> dist(0, 1000);
    std::generate(expected_keys.begin(), expected_keys.end(), [&] { return static_cast<KeyT>(dist(gen)); });
    TestUtils::generate_arithmetic_data(expected_values.data(), size, 7);

    TestUtils::usm_data_transfer<sycl::usm::alloc::device, KeyT> keys(q, expected_keys.begin(), expected_keys.end());
    TestUtils::usm_data_transfer<sycl::usm::alloc::device, ValueT> values(q, expected_values.begin(),
                                                                          expected_values.end());

    auto expected_first = oneapi::dpl::make_zip_iterator(std::begin(expected_keys), std::begin(expected_values));
    std::stable_sort(expected_first, expected_first + size, CompareKey<IsAscending>{});

    kt_ns::radix_sort_by_key<IsAscending, RadixBits>(q, keys.get_data(), keys.get_data() + size, values.get_data(),
                                                     param)
        .wait();

    std::vector<KeyT> actual_keys(size);
    std::vector<ValueT> actual_values(size);
    keys.retrieve_data(actual_keys.begin());
    values.retrieve_data(actual_values.begin());

    std::string msg = "wrong results with constrained range [0,1000] by key, n: " + std::to_string(size);
    EXPECT_EQ_N(expected_keys.begin(), actual_keys.begin(), size, (msg + " (keys)").c_str());
    EXPECT_EQ_N(expected_values.begin(), actual_values.begin(), size, (msg + " (values)").c_str());
}

int main()
{
    bool run_test = false;
#if TEST_SYCL_RADIX_SORT_KT_AVAILABLE || defined(TEST_KT_BACKEND_ESIMD)
    constexpr oneapi::dpl::experimental::kt::kernel_param<TEST_DATA_PER_WORK_ITEM, TEST_WORK_GROUP_SIZE> params;
    auto q = TestUtils::get_test_queue();
    run_test = can_run_test<decltype(params), TEST_KEY_TYPE, TEST_VALUE_TYPE>(q, params);

    if (run_test)
    {
        try
        {
            for (auto size : sort_sizes)
            {
                test_usm<TEST_KEY_TYPE, TEST_VALUE_TYPE, Ascending, TestRadixBits, sycl::usm::alloc::shared>(
                    q, size, TestUtils::create_new_kernel_param_idx<0>(params));
                test_usm<TEST_KEY_TYPE, TEST_VALUE_TYPE, Descending, TestRadixBits, sycl::usm::alloc::shared>(
                    q, size, TestUtils::create_new_kernel_param_idx<1>(params));
                test_sycl_buffer<TEST_KEY_TYPE, TEST_VALUE_TYPE, Ascending, TestRadixBits>(
                    q, size, TestUtils::create_new_kernel_param_idx<2>(params));
                test_sycl_buffer<TEST_KEY_TYPE, TEST_VALUE_TYPE, Descending, TestRadixBits>(
                    q, size, TestUtils::create_new_kernel_param_idx<3>(params));
            }
            if constexpr (std::is_floating_point_v<TEST_KEY_TYPE> || std::is_same_v<TEST_KEY_TYPE, sycl::half>)
            {
                test_negative_zero_stability<TEST_KEY_TYPE, TEST_VALUE_TYPE, Ascending, TestRadixBits>(
                    q, 64, TestUtils::create_new_kernel_param_idx<4>(params));
                test_negative_zero_stability<TEST_KEY_TYPE, TEST_VALUE_TYPE, Descending, TestRadixBits>(
                    q, 64, TestUtils::create_new_kernel_param_idx<5>(params));
                test_negative_zero_stability<TEST_KEY_TYPE, TEST_VALUE_TYPE, Ascending, TestRadixBits>(
                    q, 10000, TestUtils::create_new_kernel_param_idx<6>(params));
            }

            // Constrained-range tests to exercise single-bin optimization (only for key types >= 32 bits)
            if constexpr (std::is_integral_v<TEST_KEY_TYPE> && sizeof(TEST_KEY_TYPE) >= 4)
            {
                for (auto size : {std::size_t(1000), std::size_t(67543), std::size_t(100'000)})
                {
                    test_constrained_range_by_key<TEST_KEY_TYPE, TEST_VALUE_TYPE, Ascending, TestRadixBits>(
                        q, size, TestUtils::create_new_kernel_param_idx<4>(params));
                    test_constrained_range_by_key<TEST_KEY_TYPE, TEST_VALUE_TYPE, Descending, TestRadixBits>(
                        q, size, TestUtils::create_new_kernel_param_idx<5>(params));
                }
            }
        }
        catch (const ::std::exception& exc)
        {
            std::cerr << "Exception: " << exc.what() << std::endl;
            return EXIT_FAILURE;
        }
    }
#endif

    return TestUtils::done(run_test);
}
