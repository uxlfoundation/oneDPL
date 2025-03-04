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
        oneapi::dpl::stable_sort(std::forward<Policy>(exec), iter, iter, TestUtils::non_const(std::less<T>()));
    }
};

int main()
{
    SortTestConfig cfg;
    cfg.is_stable = true;
    cfg.test_usm_device = true;
    std::vector<std::size_t> sizes = test_sizes(TestUtils::max_n);

#if !TEST_ONLY_HETERO_POLICIES
    test_sort<TestUtils::float32_t>(SortTestConfig{cfg, "float, host, custom less comparator"}, sizes, Host{},
                                    Converter<TestUtils::float32_t>{}, ConstLess{});
    test_sort<std::int16_t>(SortTestConfig{cfg, "int16_t, host, non-const custom less comparator"}, sizes, Host{},
                            Converter<std::int16_t>{}, NonConstLess{});
    test_sort<std::uint32_t>(SortTestConfig{cfg, "uint32_t, host, std::greater"}, sizes, Host{},
                             Converter<std::uint32_t>{}, std::greater{});

    auto paranoid_key_converter = [](size_t k, size_t val) { return ParanoidKey(k, val, TestUtils::OddTag()); };
    test_sort<ParanoidKey>(SortTestConfig{cfg, "ParanoidKey, host"}, sizes, Host{},
                           paranoid_key_converter, KeyCompare(TestUtils::OddTag()));

    TestUtils::test_algo_basic_single<int32_t>(TestUtils::run_for_rnd<test_non_const<int32_t>>());
#endif

#if TEST_DPCPP_BACKEND_PRESENT
    test_sort<TestUtils::float32_t>(SortTestConfig{cfg, "float, device, custom less comparator"}, sizes, Device<0>{},
                                    Converter<TestUtils::float32_t>{}, ConstLess{});

    // Test merge-path sort specialization for large sizes, see __get_starting_size_limit_for_large_submitter
    std::vector<std::size_t> extended_sizes = test_sizes(8'000'000);
    test_sort<std::uint16_t>(SortTestConfig{cfg, "uint16_t, device, custom greater comparator"}, extended_sizes,
                             Device<1>{}, Converter<std::uint16_t>{}, ConstGreater{});
#if __SYCL_UNNAMED_LAMBDA__
    // Test potentially clashing naming for radix sort descending / ascending with minimal timing impact
    test_default_name_gen(SortTestConfig{cfg, "default name generation"});
#endif
    // TODO: add a test for stability
#endif

    return TestUtils::done();
}
