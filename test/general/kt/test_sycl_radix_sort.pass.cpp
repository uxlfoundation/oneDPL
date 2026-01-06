// -*- C++ -*-
//===-- test_sycl_radix_sort.pass.cpp ------------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "support/test_config.h"

#include <oneapi/dpl/experimental/kernel_templates>
#if _ENABLE_RANGES_TESTING
#    include <oneapi/dpl/ranges>
#endif

#include <vector>
#include <string>
#include <cstdint>
#include <cstdlib>

#include <oneapi/dpl/experimental/kt/sycl_radix_sort.h>
#include <oneapi/dpl/algorithm>
#include <vector>
#include <iostream>

#include "support/utils.h"

#define TEST_DATA_PER_WORK_ITEM 32
#define TEST_WORK_GROUP_SIZE 128


template <typename T, typename KernelParam>
bool
test_basic_sort(sycl::queue& q, std::size_t n, KernelParam params)
{
    std::vector<T> input(n);
    std::vector<T> expected(n);

    // Fill with random data
    std::generate(input.begin(), input.end(), [i = 0]() mutable {
        return static_cast<T>((i++ * 7) % 100);
    });

    expected = input;
    std::sort(expected.begin(), expected.end());

    // Test with sycl_iterator (in-place)
    {
        sycl::buffer<T> buf(input.data(), sycl::range<1>(n));
        auto policy = oneapi::dpl::execution::make_device_policy(q);
        auto begin = oneapi::dpl::begin(buf);
        auto end = oneapi::dpl::end(buf);

        auto event = oneapi::dpl::experimental::kt::gpu::__sycl::onesweep_sort(q, begin, end, params);
        event.wait();

        // Verify
        auto acc = buf.get_host_access();
        for (std::size_t i = 0; i < n; ++i)
        {
            if (acc[i] != expected[i])
            {
                std::cout << "FAILED at index " << i << ": got " << acc[i]
                         << ", expected " << expected[i] << std::endl;
                return false;
            }
        }
    }

    return true;
}

template <typename T, typename KernelParam>
bool
test_out_of_place_sort(sycl::queue& q, std::size_t n, KernelParam params)
{
    std::vector<T> input(n);
    std::vector<T> output(n);
    std::vector<T> expected(n);

    std::generate(input.begin(), input.end(), [i = 0]() mutable {
        return static_cast<T>((i++ * 13) % 100);
    });

    expected = input;
    std::sort(expected.begin(), expected.end());

    // Test out-of-place sort
    {
        sycl::buffer<T> in_buf(input.data(), sycl::range<1>(n));
        sycl::buffer<T> out_buf(output.data(), sycl::range<1>(n));

        auto in_begin = oneapi::dpl::begin(in_buf);
        auto in_end = oneapi::dpl::end(in_buf);
        auto out_begin = oneapi::dpl::begin(out_buf);

        auto event = oneapi::dpl::experimental::kt::gpu::__sycl::onesweep_sort(
            q, in_begin, in_end, out_begin, params);
        event.wait();

        // Verify output
        auto acc = out_buf.get_host_access();
        for (std::size_t i = 0; i < n; ++i)
        {
            if (acc[i] != expected[i])
            {
                std::cout << "Out-of-place FAILED at index " << i << ": got " << acc[i]
                         << ", expected " << expected[i] << std::endl;
                return false;
            }
        }

        // Verify input unchanged
        auto in_acc = in_buf.get_host_access();
        for (std::size_t i = 0; i < n; ++i)
        {
            if (in_acc[i] != input[i])
            {
                std::cout << "Input modified at index " << i << std::endl;
                return false;
            }
        }
    }

    return true;
}

int
main()
{
    constexpr oneapi::dpl::experimental::kt::kernel_param<TEST_DATA_PER_WORK_ITEM, TEST_WORK_GROUP_SIZE> params;
    sycl::queue q{sycl::gpu_selector_v};

    std::cout << "Running on: " << q.get_device().get_info<sycl::info::device::name>() << std::endl;

    bool passed = true;

    // Test various sizes
    std::vector<std::size_t> sizes = {100, 1000, 10000, 100000};

    for (auto n : sizes)
    {
        std::cout << "\nTesting size " << n << "..." << std::endl;

        // Test uint32_t
        std::cout << "  uint32_t in-place: ";
        if (test_basic_sort<std::uint32_t>(q, n, params))
            std::cout << "PASSED" << std::endl;
        else
        {
            std::cout << "FAILED" << std::endl;
            passed = false;
        }

        std::cout << "  uint32_t out-of-place: ";
        if (test_out_of_place_sort<std::uint32_t>(q, n, params))
            std::cout << "PASSED" << std::endl;
        else
        {
            std::cout << "FAILED" << std::endl;
            passed = false;
        }

        // Test int32_t (signed)
        std::cout << "  int32_t in-place: ";
        if (test_basic_sort<std::int32_t>(q, n, params))
            std::cout << "PASSED" << std::endl;
        else
        {
            std::cout << "FAILED" << std::endl;
            passed = false;
        }

        // Test float
        std::cout << "  float in-place: ";
        if (test_basic_sort<float>(q, n, params))
            std::cout << "PASSED" << std::endl;
        else
        {
            std::cout << "FAILED" << std::endl;
            passed = false;
        }
    }

    std::cout << "\n" << (passed ? "All tests PASSED" : "Some tests FAILED") << std::endl;

    return passed ? 0 : 1;
}
