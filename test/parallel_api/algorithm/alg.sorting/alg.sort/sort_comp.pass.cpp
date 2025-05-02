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

template <typename T>
struct test_non_const
{
    template <typename Policy, typename Iterator>
    void
    operator()(Policy&& exec, Iterator iter)
    {
        oneapi::dpl::sort(std::forward<Policy>(exec), iter, iter, TestUtils::non_const(std::less<T>()));
    }
};

int main()
{
    SortTestConfig cfg;
    cfg.is_stable = false;
    cfg.test_usm_shared = true;
    std::vector<std::size_t> sizes = test_sizes(TestUtils::max_n);

#if !TEST_ONLY_HETERO_POLICIES
    test_sort<TestUtils::float32_t>(SortTestConfig{cfg, "float, host, custom greater"}, sizes, Host{},
                                    Converter<TestUtils::float32_t>{}, ConstGreater{});
    test_sort<std::uint16_t>(SortTestConfig{cfg, "uint16_t, host, non-const custom less"}, sizes, Host{},
                                     Converter<std::uint16_t>{}, NonConstLess{});

    test_sort<ParanoidKey>(SortTestConfig{cfg, "ParanoidKey, host"}, sizes, Host{},
                           [](size_t k, size_t val) { return ParanoidKey(k, val, TestUtils::OddTag()); },
                           KeyCompare(TestUtils::OddTag()));

    TestUtils::test_algo_basic_single<std::int32_t>(TestUtils::run_for_rnd<test_non_const<std::int32_t>>());
#endif

#if TEST_DPCPP_BACKEND_PRESENT
    test_sort<TestUtils::float32_t>(SortTestConfig{cfg, "float, device, custom greater"}, sizes, Device<0>{},
                                    Converter<TestUtils::float32_t>{}, ConstGreater{});
    test_sort<std::uint16_t>(SortTestConfig{cfg, "uint16_t, device, non-const custom less"}, sizes, Device<1>{},
                             Converter<std::uint16_t>{}, NonConstLess{});

    // Check radix-sort with to have a higher chance to hit synchronization issues if any
    sizes.push_back(8'000'000);
    test_sort<std::int32_t>(SortTestConfig{cfg, "int32_t, device, std::less"}, sizes, Device<2>{},
                            Converter<std::int32_t>{}, std::less{});
#if __SYCL_UNNAMED_LAMBDA__
    // Test potentially clashing naming for radix sort descending / ascending with minimal timing impact
    test_default_name_gen(SortTestConfig{cfg, "default name generation"});
#endif
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done();
}
