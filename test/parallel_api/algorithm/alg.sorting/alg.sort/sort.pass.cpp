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

#include "support/utils_sort.h" // Umbrella for all needed headers

int main()
{
    SortTestConfig cfg;
    cfg.is_stable = false;
    cfg.test_usm_device = true;
    const std::vector<std::size_t> sizes = test_sizes(TestUtils::max_n);
    const std::vector<std::size_t> small_sizes = test_sizes(10'000);

#if !TEST_ONLY_HETERO_POLICIES
    test_sort<TestUtils::float32_t>(SortTestConfig{cfg, "float, host"}, sizes, Host{},
                                    Converter<TestUtils::float32_t>{});
    test_sort<unsigned char>(SortTestConfig{cfg, "unsigned char, host"}, sizes, Host{},
                             Converter<unsigned char>{});
#endif

#if TEST_DPCPP_BACKEND_PRESENT
    test_sort<TestUtils::float32_t>(SortTestConfig{cfg, "float, device"}, sizes, Device<0>{},
                                    Converter<TestUtils::float32_t>{});

    auto half_converter = [](size_t /*index*/, size_t val) {
        return TestUtils::sycl_half_convert(std::uint16_t(val & 0xFFFFu));
    };

    // Test radix-sort bit conversions
    test_sort<std::uint8_t>(SortTestConfig{cfg, "uint8_t, device"}, small_sizes, Device<1>{},
                            Converter<std::uint8_t>{});
    test_sort<std::int16_t>(SortTestConfig{cfg, "int16_t, device"}, small_sizes, Device<2>{},
                            Converter<std::int16_t>{});
    test_sort<std::uint32_t>(SortTestConfig{cfg, "uint32_t, device"}, small_sizes, Device<3>{},
                             Converter<std::uint32_t>{});
    test_sort<std::int64_t>(SortTestConfig{cfg, "int64_t, device"}, small_sizes, Device<4>{},
                            Converter<std::int64_t>{});
    test_sort<TestUtils::float64_t>(SortTestConfig{cfg, "float64_t, device"}, small_sizes, Device<5>{},
                                    Converter<TestUtils::float64_t>{});
    test_sort<sycl::half>(SortTestConfig{cfg, "sycl::half, device"}, small_sizes, Device<6>{}, half_converter);
    // TODO: add a test for a MoveConstructible only type
#endif

    return TestUtils::done();
}
