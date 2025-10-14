// -*- C++ -*-
//===-- esimd_radix_sort_by_key.cpp -----------------------------------------===//
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

#include "esimd_radix_sort_utils.h"

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
        oneapi::dpl::experimental::kt::gpu::esimd::radix_sort_by_key<isAscending, RadixBits>(q, keys, values, param).wait();
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

    oneapi::dpl::experimental::kt::gpu::esimd::radix_sort_by_key<isAscending, RadixBits>(
        q, keys.get_data(), keys.get_data() + size, values.get_data(), param).wait();

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

int main()
{
    constexpr oneapi::dpl::experimental::kt::kernel_param<TEST_DATA_PER_WORK_ITEM, TEST_WORK_GROUP_SIZE> params;
    auto q = TestUtils::get_test_queue();
    bool run_test = can_run_test<decltype(params), TEST_KEY_TYPE, TEST_VALUE_TYPE>(q, params);

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
        }
        catch (const ::std::exception& exc)
        {
            std::cerr << "Exception: " << exc.what() << std::endl;
            return EXIT_FAILURE;
        }
    }

    return TestUtils::done(run_test);
}
