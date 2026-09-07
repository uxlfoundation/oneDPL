// -*- C++ -*-
//===-- unique_segmented.pass.cpp -----------------------------------------===//
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

// Test for unique over the device backend's segmented compaction path.
//
// The device unique pattern stages its output through a temporary bounded at _ONEDPL_COMPACTION_SEGMENT_SIZE_BYTES and
// compacts the input one segment at a time whenever the input does not fit in a single segment. The default bound is
// far larger than any input a test can afford, so the override below shrinks it to a few elements. The override has to
// be visible before any oneDPL header, and is a variable rather than a literal so that one translation unit can sweep
// several segment sizes.

#include <cstddef>

namespace segment_size_override
{
std::size_t bytes = 64 * 1024 * 1024;
}
#define _ONEDPL_COMPACTION_SEGMENT_SIZE_BYTES (::segment_size_override::bytes)

#include "support/test_config.h"

#include _PSTL_TEST_HEADER(execution)
#include _PSTL_TEST_HEADER(algorithm)

#include "support/utils.h"

#if TEST_DPCPP_BACKEND_PRESENT
#    include "support/utils_sycl.h"
#    include "support/sycl_alloc_utils.h"
#endif

#include <algorithm>
#include <cstdint>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

#if TEST_DPCPP_BACKEND_PRESENT

template <typename... Name>
struct unique_segmented_kernel;

// Where the input lives; also part of the kernel name, so each path gets its own kernel.
struct host_iterators_tag
{
    static constexpr const char* name = "host iterators";
};
struct buffer_tag
{
    static constexpr const char* name = "sycl::buffer";
};
struct usm_shared_tag
{
    static constexpr const char* name = "USM shared";
};
struct usm_device_tag
{
    static constexpr const char* name = "USM device";
};

enum class pattern
{
    all_equal,
    all_distinct,
    alternating,
    runs_of_2,
    runs_of_3,
    straddling_runs,
    random_few_values
};

constexpr pattern all_patterns[] = {pattern::all_equal, pattern::all_distinct,    pattern::alternating,
                                    pattern::runs_of_2, pattern::runs_of_3,       pattern::straddling_runs,
                                    pattern::random_few_values};

const char*
pattern_name(pattern p)
{
    switch (p)
    {
    case pattern::all_equal:
        return "all_equal";
    case pattern::all_distinct:
        return "all_distinct";
    case pattern::alternating:
        return "alternating";
    case pattern::runs_of_2:
        return "runs_of_2";
    case pattern::runs_of_3:
        return "runs_of_3";
    case pattern::straddling_runs:
        return "straddling_runs";
    default:
        return "random_few_values";
    }
}

template <typename T>
std::vector<T>
make_input(pattern p, std::size_t n, std::size_t segment_size)
{
    std::vector<T> v(n);
    std::mt19937 gen(std::uint32_t(n * 1024 + segment_size));
    std::uniform_int_distribution<int> few_values(0, 2);

    for (std::size_t i = 0; i < n; ++i)
    {
        switch (p)
        {
        case pattern::all_equal:
            v[i] = T(1);
            break;
        case pattern::all_distinct:
            v[i] = T(i);
            break;
        case pattern::alternating:
            v[i] = T(i % 2);
            break;
        case pattern::runs_of_2:
            v[i] = T(i / 2);
            break;
        case pattern::runs_of_3:
            v[i] = T(i / 3);
            break;
        // Repeats the element before every segment boundary, so a run of equal elements straddles each boundary.
        case pattern::straddling_runs:
            v[i] = T(i - ((i > 0 && i % segment_size == 0) ? 1 : 0));
            break;
        default:
            v[i] = T(few_values(gen));
            break;
        }
    }
    return v;
}

// std::unique compares against the last kept element while the device pattern compares against the original
// neighbour, so the two agree only for a transitive predicate. Bucketing by a function of the value is one.
struct equal_bucket
{
    template <typename T>
    bool
    operator()(const T& a, const T& b) const
    {
        return std::int64_t(a) / 4 == std::int64_t(b) / 4;
    }
};

struct unique_default
{
    static constexpr const char* name = "unique";

    template <typename Policy, typename It>
    It
    operator()(Policy&& exec, It first, It last) const
    {
        return oneapi::dpl::unique(std::forward<Policy>(exec), first, last);
    }
    template <typename It>
    It
    reference(It first, It last) const
    {
        return std::unique(first, last);
    }
};

struct unique_with_predicate
{
    static constexpr const char* name = "unique with predicate";

    template <typename Policy, typename It>
    It
    operator()(Policy&& exec, It first, It last) const
    {
        return oneapi::dpl::unique(std::forward<Policy>(exec), first, last, equal_bucket{});
    }
    template <typename It>
    It
    reference(It first, It last) const
    {
        return std::unique(first, last, equal_bucket{});
    }
};

template <typename T, typename Policy, typename Algo, typename MemTag>
void
run_case(Policy&& exec, Algo algo, MemTag, std::size_t n, std::size_t segment_size, pattern p)
{
    using kernel_name = unique_segmented_kernel<T, Algo, MemTag>;

    const std::vector<T> input = make_input<T>(p, n, segment_size);

    std::vector<T> expected(input);
    const std::size_t expected_n = std::size_t(algo.reference(expected.begin(), expected.end()) - expected.begin());

    std::vector<T> actual(input);
    std::size_t actual_n = 0;

    segment_size_override::bytes = segment_size * sizeof(T);

    if constexpr (std::is_same_v<MemTag, host_iterators_tag>)
    {
        auto first = actual.begin();
        actual_n = std::size_t(algo(CLONE_TEST_POLICY_NAME(exec, kernel_name), first, first + n) - first);
    }
    else if constexpr (std::is_same_v<MemTag, buffer_tag>)
    {
        // A sycl::buffer range must not be empty, so the n == 0 case gets one unused element.
        std::vector<T> storage(actual);
        storage.resize(std::max<std::size_t>(n, 1));
        {
            sycl::buffer<T> buf(storage.data(), sycl::range<1>(storage.size()));
            auto first = oneapi::dpl::begin(buf);
            actual_n = std::size_t(algo(CLONE_TEST_POLICY_NAME(exec, kernel_name), first, first + n) - first);
        }
        std::copy_n(storage.begin(), n, actual.begin());
    }
    else
    {
        constexpr sycl::usm::alloc alloc_type =
            std::is_same_v<MemTag, usm_shared_tag> ? sycl::usm::alloc::shared : sycl::usm::alloc::device;
        TestUtils::usm_data_transfer<alloc_type, T> dt_helper(exec.queue(), actual.begin(), n);
        T* first = dt_helper.get_data();
        actual_n = std::size_t(algo(CLONE_TEST_POLICY_NAME(exec, kernel_name), first, first + n) - first);
        dt_helper.retrieve_data(actual.begin());
    }

    const std::string msg = std::string("wrong effect from ") + Algo::name + " over " + MemTag::name + ": n = " +
                            std::to_string(n) + ", segment size = " + std::to_string(segment_size) + " element(s), " +
                            pattern_name(p) + " input";

    EXPECT_EQ(expected_n, actual_n, msg.c_str());
    // Only the returned prefix is checked: the standard leaves the values past the returned iterator unspecified,
    // so nothing may be asserted about them.
    EXPECT_EQ_N(expected.begin(), actual.begin(), expected_n, msg.c_str());
}

// Sizes around the powers of two and around the segment boundaries.
std::vector<std::size_t>
test_sizes(std::size_t s)
{
    std::vector<std::size_t> sizes = {0, 1, 2, 3, 4, 5, 6, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 100};
    for (std::size_t n : {s - 1, s, s + 1, 2 * s, 2 * s + 1, 3 * s})
        if (std::find(sizes.begin(), sizes.end(), n) == sizes.end())
            sizes.push_back(n);
    return sizes;
}

template <typename Policy>
void
test(Policy&& exec)
{
    const bool fp64 = TestUtils::has_types_support<TestUtils::float64_t>(exec.queue().get_device());
    if (!fp64)
        TestUtils::unsupported_types_notifier(exec.queue().get_device());

    // (segment size, n, input pattern) is the surface the segmented loop depends on; sweep it fully on one path.
    for (std::size_t s : {std::size_t(1), std::size_t(2), std::size_t(3)})
        for (std::size_t n : test_sizes(s))
            for (pattern p : all_patterns)
                run_case<std::int32_t>(exec, unique_default{}, buffer_tag{}, n, s, p);

    // The remaining paths, the predicate overload and further value types are orthogonal to that surface, so they
    // only get a slice of it. std::int64_t additionally varies sizeof(T), which the byte bound is divided by.
    for (std::size_t s : {std::size_t(1), std::size_t(2), std::size_t(3)})
        for (std::size_t n : {0, 1, 2, 7, 16, 33, 100})
            for (pattern p : {pattern::runs_of_2, pattern::straddling_runs, pattern::random_few_values})
            {
                run_case<std::int32_t>(exec, unique_default{}, host_iterators_tag{}, n, s, p);
                run_case<std::int32_t>(exec, unique_default{}, usm_shared_tag{}, n, s, p);
                run_case<std::int32_t>(exec, unique_default{}, usm_device_tag{}, n, s, p);
                run_case<std::int32_t>(exec, unique_with_predicate{}, buffer_tag{}, n, s, p);
                run_case<std::int32_t>(exec, unique_with_predicate{}, usm_device_tag{}, n, s, p);
                run_case<std::int64_t>(exec, unique_default{}, buffer_tag{}, n, s, p);
                if (fp64)
                    run_case<TestUtils::float64_t>(exec, unique_default{}, buffer_tag{}, n, s, p);
            }
}
#endif // TEST_DPCPP_BACKEND_PRESENT

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    test(TestUtils::get_dpcpp_test_policy());
#endif

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
