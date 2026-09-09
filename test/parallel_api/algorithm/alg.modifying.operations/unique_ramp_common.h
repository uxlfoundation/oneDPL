// -*- C++ -*-
//===-- unique_ramp_common.h ----------------------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// DIAGNOSTIC BUILD ONLY, DO NOT MERGE. Shared body of the five unique_ramp_*.pass tests, which localize the
// SEGFAULT of unique_segmented.pass on the Windows debug GPU configurations (CI runs 34074196751 and
// 34313887977). That sweep varies n, the segment size, the memory path, the value type, the input pattern and
// the predicate all at once; its crash coordinate moved between jobs but landed 11 times out of 12 on the
// same call shape — the predicate overload over a sycl::buffer at one element per segment — so no single axis
// is implicated yet. Each test here varies exactly one thing against unique_ramp_buf:
//
//   unique_ramp_buf     plain unique, sycl::buffer, runs of 2   the reference ramp, 1 -> n segments
//   unique_ramp_usm     ... over a USM device pointer           isolates the memory path
//   unique_ramp_pred    ... with the predicate overload         the shape 11 of 12 crashes landed on
//   unique_ramp_runs8   ... plain unique, runs of 8             the predicate's 8:1 drop rate without it
//   unique_ramp_repeat  ... 1,000 calls at 16 segments          isolates anything accumulating per call
//
// A threshold in the segment count bounds the input size a real user may pass, since the shipping bound gives
// n/16M segments. pred crashing while runs8 does not implicates the overload's own code; both crashing
// implicates the drop rate, i.e. how far the output position lags the input position.
//
// Each test is its own process, so one crashing does not hide the others.

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
#include <iostream>
#include <type_traits>
#include <vector>

// Without this the bound would be the shipping 64 MiB and every case below would run one segment, so a green
// run would prove nothing. ramp_arm_probe.cpp checks the assert does fire when the override is removed.
static_assert(_ONEDPL_COMPACTION_SEGMENT_SIZE_FORCED == 1, "the segment bound override did not reach the library");

#if TEST_DPCPP_BACKEND_PRESENT

template <typename... Name>
struct unique_ramp_kernel;

struct buffer_tag
{
    static constexpr const char* name = "sycl::buffer";
};
struct usm_device_tag
{
    static constexpr const char* name = "USM device";
};

using value_type = std::int32_t;

struct plain_unique
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

// Bucketing by a function of the value is transitive, so std::unique agrees with the device pattern even
// though the two compare against different elements. Matches the crashing sweep's predicate.
struct equal_bucket
{
    bool
    operator()(const value_type& a, const value_type& b) const
    {
        return std::int64_t(a) / 4 == std::int64_t(b) / 4;
    }
};

struct predicate_unique
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

// run_length 2 keeps every other element; 8 keeps one in eight, which is what equal_bucket does to run_length 2.
std::vector<value_type>
make_input(std::size_t n, std::size_t run_length)
{
    std::vector<value_type> v(n);
    for (std::size_t i = 0; i < n; ++i)
        v[i] = value_type(i / run_length);
    return v;
}

// One oneapi::dpl::unique over n elements with the segment bound set so the pattern runs exactly
// `segments` iterations. `segments` must divide n.
template <typename MemTag, typename Algo, typename Policy>
void
run_one(Policy&& exec, Algo algo, const std::vector<value_type>& input, std::size_t segments)
{
    using kernel_name = unique_ramp_kernel<MemTag, Algo>;

    const std::size_t n = input.size();
    ::segment_size_override::bytes = (n / segments) * sizeof(value_type);

    std::vector<value_type> expected(input);
    const std::size_t expected_n = std::size_t(algo.reference(expected.begin(), expected.end()) - expected.begin());

    std::vector<value_type> actual(input);
    std::size_t actual_n = 0;

    if constexpr (std::is_same_v<MemTag, buffer_tag>)
    {
        sycl::buffer<value_type> buf(actual.data(), sycl::range<1>(n));
        auto first = oneapi::dpl::begin(buf);
        actual_n = std::size_t(algo(CLONE_TEST_POLICY_NAME(exec, kernel_name), first, first + n) - first);
    }
    else
    {
        TestUtils::usm_data_transfer<sycl::usm::alloc::device, value_type> dt_helper(exec.queue(), actual.begin(), n);
        value_type* first = dt_helper.get_data();
        actual_n = std::size_t(algo(CLONE_TEST_POLICY_NAME(exec, kernel_name), first, first + n) - first);
        dt_helper.retrieve_data(actual.begin());
    }

    EXPECT_EQ(expected_n, actual_n, "wrong number of survivors");
    EXPECT_EQ_N(expected.begin(), actual.begin(), expected_n, "wrong survivors");
}

// Segment count 1 (the shipping fast path) up to n (one element per segment), at two fixed sizes. Within a
// size, everything but the segment count is held constant.
template <typename MemTag, typename Algo, typename Policy>
void
run_ramp(Policy&& exec, Algo algo, std::size_t run_length)
{
    for (std::size_t n : {std::size_t(1024), std::size_t(4096)})
    {
        const std::vector<value_type> input = make_input(n, run_length);
        for (std::size_t segments = 1; segments <= n; segments *= 2)
        {
            std::cout << Algo::name << " | " << MemTag::name << " | runs=" << run_length << " n=" << n
                      << " segments=" << segments << " s=" << n / segments << std::endl;
            run_one<MemTag>(exec, algo, input, segments);
            std::cout << "    ok" << std::endl;
        }
    }
}

// A modest segment count, many times over. Isolates anything that accumulates across calls.
template <typename MemTag, typename Algo, typename Policy>
void
run_repeat(Policy&& exec, Algo algo, std::size_t run_length, std::size_t iterations, std::size_t segments)
{
    const std::size_t n = 4096;
    const std::vector<value_type> input = make_input(n, run_length);
    std::cout << Algo::name << " | " << MemTag::name << " | n=" << n << " segments=" << segments << " x"
              << iterations << std::endl;
    for (std::size_t i = 0; i < iterations; ++i)
    {
        if (i % 50 == 0)
            std::cout << "  call " << i << std::endl;
        run_one<MemTag>(exec, algo, input, segments);
    }
    std::cout << "    ok" << std::endl;
}
#endif // TEST_DPCPP_BACKEND_PRESENT
