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
    cfg.is_stable = true;
    cfg.test_usm_shared = true;
    std::vector<std::size_t> sizes = test_sizes(TestUtils::max_n);

#if !TEST_ONLY_HETERO_POLICIES
    test_sort<TestUtils::float32_t>(SortTestConfig{cfg, "float, host"}, sizes, Host{},
                                    Converter<TestUtils::float32_t>{});
    test_sort<std::int64_t>(SortTestConfig{cfg, "int64_t, host"}, sizes, Host{}, Converter<std::int64_t>{});
    // TODO: add a test for stability
#endif

#if TEST_DPCPP_BACKEND_PRESENT
    test_sort<TestUtils::float32_t>(SortTestConfig{cfg, "float, device"}, sizes, Device<0>{},
                                    Converter<TestUtils::float32_t>{});
    test_sort<std::int64_t>(SortTestConfig{cfg, "int64_t, device"}, sizes, Device<1>{}, Converter<std::int64_t>{});
    // TODO: add a test for stability
#endif

    return TestUtils::done();
}
