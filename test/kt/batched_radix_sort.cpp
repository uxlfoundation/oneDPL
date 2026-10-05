// -*- C++ -*-
//===-- batched_radix_sort.cpp --------------------------------------------===//
//
// Copyright (C) UXL Foundation Contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../support/test_config.h"

#include <oneapi/dpl/experimental/kernel_templates>
#include <oneapi/dpl/iterator>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#if __has_include(<sycl/sycl.hpp>)
#    include <sycl/sycl.hpp>
#else
#    include <CL/sycl.hpp>
#endif

#include "../support/utils.h"
#include "../support/sycl_alloc_utils.h"

#include "radix_sort_utils.h"

namespace kt = oneapi::dpl::experimental::kt;

#ifndef TEST_VALUE_TYPE
#    define TEST_KEYS_ONLY 1
using ValueT = std::uint32_t; // unused by the sorts, only by the reference
#else
using ValueT = TEST_VALUE_TYPE;
#endif
using KeyT = TEST_KEY_TYPE;

#ifdef TEST_RADIX_BITS
constexpr std::uint8_t BatchedRadixBits = TEST_RADIX_BITS;
#else
constexpr std::uint8_t BatchedRadixBits = TestRadixBits;
#endif
// Only onesweep (8 radix bits) sorts segments larger than a tile
constexpr bool OneWorkGroupOnly = BatchedRadixBits != 8;

enum class DataMode
{
    usm_iterators,
    usm_ranges,
    buffer_iterators,
    buffer_ranges
};

const char*
mode_name(DataMode mode)
{
    switch (mode)
    {
    case DataMode::usm_iterators:
        return "USM iterators";
    case DataMode::usm_ranges:
        return "USM ranges";
    case DataMode::buffer_iterators:
        return "sycl::buffer iterators";
    default:
        return "sycl::buffer ranges";
    }
}

enum class DataPattern
{
    random,
    // Few distinct keys, values record the original position within the segment to check stability
    few_distinct,
    // A single key: every element goes to the last bins together with the padding of partial tiles
    all_equal,
    // Keys decrease across the whole input, so every segment must move its elements in the opposite direction of its
    // neighbors: elements leaking across a segment boundary change the result
    reversed,
    // -0.0, +0.0 (equal for the sort) and infinities
    float_specials
};

const char*
pattern_name(DataPattern pattern)
{
    switch (pattern)
    {
    case DataPattern::random:
        return "random";
    case DataPattern::few_distinct:
        return "few distinct";
    case DataPattern::all_equal:
        return "all equal";
    case DataPattern::reversed:
        return "reversed";
    default:
        return "float specials";
    }
}

constexpr bool FloatingKey = !std::is_integral_v<KeyT>;

// Floating-point keys are compared by bit pattern: the sort moves keys unchanged, and TestUtils compares floating-point
// values with a tolerance that fails for infinities and does not distinguish -0.0 from +0.0
using KeyBitsT =
    std::conditional_t<sizeof(KeyT) == 1, std::uint8_t,
                       std::conditional_t<sizeof(KeyT) == 2, std::uint16_t,
                                          std::conditional_t<sizeof(KeyT) == 4, std::uint32_t, std::uint64_t>>>;

std::vector<KeyBitsT>
key_bits(const std::vector<KeyT>& keys)
{
    std::vector<KeyBitsT> bits(keys.size());
    std::memcpy(bits.data(), keys.data(), keys.size() * sizeof(KeyT));
    return bits;
}

void
expect_equal_keys(const std::vector<KeyT>& expected, const std::vector<KeyT>& actual, const std::string& message)
{
    if constexpr (FloatingKey)
    {
        const std::vector<KeyBitsT> expected_bits = key_bits(expected), actual_bits = key_bits(actual);
        EXPECT_EQ_N(expected_bits.begin(), actual_bits.begin(), expected.size(), message.c_str());
    }
    else
    {
        EXPECT_EQ_N(expected.begin(), actual.begin(), expected.size(), message.c_str());
    }
}

void
generate_data(std::vector<KeyT>& keys, std::vector<ValueT>& vals, std::size_t segment_size, DataPattern pattern)
{
    const std::size_t n = keys.size();
    if (pattern == DataPattern::random)
    {
        TestUtils::generate_arithmetic_data(keys.data(), n, 42 + segment_size);
        TestUtils::generate_arithmetic_data(vals.data(), n, 7);
        return;
    }
    for (std::size_t i = 0; i < n; ++i)
    {
        vals[i] = ValueT(i % segment_size);
        switch (pattern)
        {
        case DataPattern::few_distinct:
            keys[i] = KeyT((i * 7919) % 5);
            break;
        case DataPattern::all_equal:
            keys[i] = KeyT(3);
            break;
        case DataPattern::reversed:
            if constexpr (FloatingKey)
                keys[i] = KeyT(float(n - 1 - i));
            else
                keys[i] = KeyT(n - 1 - i);
            break;
        default:
            if constexpr (FloatingKey)
            {
                const float specials[] = {
                    -0.0f, 0.0f, -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
                    1.5f,  -1.5f};
                keys[i] = KeyT(specials[(i * 7919) % 6]);
            }
            break;
        }
    }
}

// The ordering of kt sorts: the radix order-preserving transformation (-0.0 == +0.0)
template <bool IsAscending>
struct KtOrder
{
    bool
    operator()(const KeyT& a, const KeyT& b) const
    {
        return oneapi::dpl::__internal::__order_preserving_cast<IsAscending>(a) <
               oneapi::dpl::__internal::__order_preserving_cast<IsAscending>(b);
    }
};

template <bool IsAscending>
void
reference_sort(std::vector<KeyT>& keys, std::vector<ValueT>& vals, std::size_t segment_size)
{
    std::vector<std::size_t> idx(segment_size);
    std::vector<KeyT> k(segment_size);
    std::vector<ValueT> v(segment_size);
    for (std::size_t first = 0; first < keys.size(); first += segment_size)
    {
        std::iota(idx.begin(), idx.end(), first);
        std::stable_sort(idx.begin(), idx.end(),
                         [&](std::size_t a, std::size_t b) { return KtOrder<IsAscending>{}(keys[a], keys[b]); });
        for (std::size_t i = 0; i < segment_size; ++i)
        {
            k[i] = keys[idx[i]];
            v[i] = vals[idx[i]];
        }
        std::copy(k.begin(), k.end(), keys.begin() + first);
        std::copy(v.begin(), v.end(), vals.begin() + first);
    }
}

// Sorts on the device with the given data passing mode. The output vectors are both input and output for
// in-place sorts.
template <bool IsAscending, bool InPlace, typename KernelParam>
void
device_sort(sycl::queue q, DataMode mode, const std::vector<KeyT>& keys_in, const std::vector<ValueT>& vals_in,
            std::vector<KeyT>& keys_out, std::vector<ValueT>& vals_out, std::vector<KeyT>& keys_in_after,
            std::vector<ValueT>& vals_in_after, std::size_t segment_size, KernelParam param)
{
    const std::size_t n = keys_in.size();
    keys_in_after = keys_in;
    vals_in_after = vals_in;
    keys_out.assign(n, KeyT{9});
    vals_out.assign(n, ValueT{9});

    if (mode == DataMode::usm_iterators || mode == DataMode::usm_ranges)
    {
        TestUtils::usm_data_transfer<sycl::usm::alloc::device, KeyT> k(q, keys_in_after.begin(), keys_in_after.end());
        TestUtils::usm_data_transfer<sycl::usm::alloc::device, ValueT> v(q, vals_in_after.begin(), vals_in_after.end());
        TestUtils::usm_data_transfer<sycl::usm::alloc::device, KeyT> ko(q, keys_out.begin(), keys_out.end());
        TestUtils::usm_data_transfer<sycl::usm::alloc::device, ValueT> vo(q, vals_out.begin(), vals_out.end());
        KeyT* kp = k.get_data();
        ValueT* vp = v.get_data();
        KeyT* kop = ko.get_data();
        ValueT* vop = vo.get_data();
        sycl::event e;
        if (mode == DataMode::usm_iterators)
        {
#if TEST_KEYS_ONLY
            if constexpr (InPlace)
                e = kt::gpu::batched_radix_sort<IsAscending, BatchedRadixBits>(q, kp, kp + n, segment_size, param);
            else
                e = kt::gpu::batched_radix_sort<IsAscending, BatchedRadixBits>(q, kp, kp + n, kop, segment_size, param);
#else
            if constexpr (InPlace)
                e = kt::gpu::batched_radix_sort_by_key<IsAscending, BatchedRadixBits>(q, kp, kp + n, vp, segment_size,
                                                                                   param);
            else
                e = kt::gpu::batched_radix_sort_by_key<IsAscending, BatchedRadixBits>(q, kp, kp + n, vp, kop, vop,
                                                                                   segment_size, param);
#endif
        }
        else
        {
            // USM ranges are passed as subrange views over the pointers
            auto kv = oneapi::dpl::experimental::ranges::views::subrange(kp, kp + n);
            auto vv = oneapi::dpl::experimental::ranges::views::subrange(vp, vp + n);
            auto kov = oneapi::dpl::experimental::ranges::views::subrange(kop, kop + n);
            auto vov = oneapi::dpl::experimental::ranges::views::subrange(vop, vop + n);
            (void)vv;
            (void)kov;
            (void)vov;
#if TEST_KEYS_ONLY
            if constexpr (InPlace)
                e = kt::gpu::batched_radix_sort<IsAscending, BatchedRadixBits>(q, kv, segment_size, param);
            else
                e = kt::gpu::batched_radix_sort<IsAscending, BatchedRadixBits>(q, kv, kov, segment_size, param);
#else
            if constexpr (InPlace)
                e = kt::gpu::batched_radix_sort_by_key<IsAscending, BatchedRadixBits>(q, kv, vv, segment_size, param);
            else
                e = kt::gpu::batched_radix_sort_by_key<IsAscending, BatchedRadixBits>(q, kv, vv, kov, vov, segment_size,
                                                                                   param);
#endif
        }
        e.wait();
        if constexpr (InPlace)
        {
            k.retrieve_data(keys_out.begin());
            v.retrieve_data(vals_out.begin());
        }
        else
        {
            ko.retrieve_data(keys_out.begin());
            vo.retrieve_data(vals_out.begin());
            k.retrieve_data(keys_in_after.begin());
            v.retrieve_data(vals_in_after.begin());
        }
    }
    else
    {
        std::vector<KeyT> kh(keys_in);
        std::vector<ValueT> vh(vals_in);
        {
            sycl::buffer<KeyT> k(kh.data(), n);
            sycl::buffer<ValueT> v(vh.data(), n);
            sycl::buffer<KeyT> ko(keys_out.data(), n);
            sycl::buffer<ValueT> vo(vals_out.data(), n);
            if (mode == DataMode::buffer_iterators)
            {
                auto kb = oneapi::dpl::begin(k);
                auto vb = oneapi::dpl::begin(v);
                auto kob = oneapi::dpl::begin(ko);
                auto vob = oneapi::dpl::begin(vo);
                (void)vb;
                (void)kob;
                (void)vob;
#if TEST_KEYS_ONLY
                if constexpr (InPlace)
                    kt::gpu::batched_radix_sort<IsAscending, BatchedRadixBits>(q, kb, kb + n, segment_size, param)
                        .wait();
                else
                    kt::gpu::batched_radix_sort<IsAscending, BatchedRadixBits>(q, kb, kb + n, kob, segment_size, param)
                        .wait();
#else
                if constexpr (InPlace)
                    kt::gpu::batched_radix_sort_by_key<IsAscending, BatchedRadixBits>(q, kb, kb + n, vb, segment_size,
                                                                                   param)
                        .wait();
                else
                    kt::gpu::batched_radix_sort_by_key<IsAscending, BatchedRadixBits>(q, kb, kb + n, vb, kob, vob,
                                                                                   segment_size, param)
                        .wait();
#endif
            }
            else
            {
#if TEST_KEYS_ONLY
                if constexpr (InPlace)
                    kt::gpu::batched_radix_sort<IsAscending, BatchedRadixBits>(q, k, segment_size, param).wait();
                else
                    kt::gpu::batched_radix_sort<IsAscending, BatchedRadixBits>(q, k, ko, segment_size, param).wait();
#else
                if constexpr (InPlace)
                    kt::gpu::batched_radix_sort_by_key<IsAscending, BatchedRadixBits>(q, k, v, segment_size, param)
                        .wait();
                else
                    kt::gpu::batched_radix_sort_by_key<IsAscending, BatchedRadixBits>(q, k, v, ko, vo, segment_size,
                                                                                      param)
                        .wait();
#endif
            }
        }
        if constexpr (InPlace)
        {
            keys_out = kh;
            vals_out = vh;
        }
        else
        {
            keys_in_after = kh;
            vals_in_after = vh;
        }
    }
}

template <bool IsAscending, bool InPlace, typename KernelParam>
void
test_case(sycl::queue q, DataMode mode, std::size_t segment_size, std::size_t segment_count, KernelParam param,
          DataPattern pattern = DataPattern::random)
{
    const std::size_t n = segment_size * segment_count;
    std::vector<KeyT> keys(n);
    std::vector<ValueT> vals(n);
    generate_data(keys, vals, segment_size, pattern);

    std::vector<KeyT> expected_keys(keys);
    std::vector<ValueT> expected_vals(vals);
    reference_sort<IsAscending>(expected_keys, expected_vals, segment_size);

    std::vector<KeyT> actual_keys, keys_in_after;
    std::vector<ValueT> actual_vals, vals_in_after;
    device_sort<IsAscending, InPlace>(q, mode, keys, vals, actual_keys, actual_vals, keys_in_after, vals_in_after,
                                      segment_size, param);

    std::ostringstream msg;
    msg << mode_name(mode) << ", in-place: " << InPlace << ", ascending: " << IsAscending
        << ", segment size: " << segment_size << ", segments: " << segment_count
        << ", dpwi: " << KernelParam::data_per_workitem << ", wgs: " << KernelParam::workgroup_size
        << ", data: " << pattern_name(pattern);
    const std::string m = msg.str();

    expect_equal_keys(expected_keys, actual_keys, "wrong keys, " + m);
#if !TEST_KEYS_ONLY
    EXPECT_EQ_N(expected_vals.begin(), actual_vals.begin(), n, ("wrong values, " + m).c_str());
#endif
    if constexpr (!InPlace)
    {
        expect_equal_keys(keys, keys_in_after, "input keys modified, " + m);
#if !TEST_KEYS_ONLY
        EXPECT_EQ_N(vals.begin(), vals_in_after.begin(), n, ("input values modified, " + m).c_str());
#endif
    }
}

// Segment sizes around the tile size (data_per_workitem * workgroup_size) and the global histogram chunk (4096), plus
// segments smaller than a tile, multi-tile segments and segments that are not multiples of the sub-group size.
// Segments that fit in a tile are owned by whole sub-groups of 32 * data_per_workitem elements, several per
// work-group, so sizes around multiples of a sub-group's elements are covered too.
std::vector<std::size_t>
segment_sizes(std::size_t tile, std::size_t sub_group_tile)
{
    std::vector<std::size_t> sizes = {
        1,        2,      7,        100,      317,          1000,           4095,         4097,
        tile - 1, tile,   tile + 1, 2 * tile, 2 * tile + 1, 3 * tile + 333, 7 * tile - 5, 50'000,
        1 << 17,  300'007,
        sub_group_tile - 1, sub_group_tile, sub_group_tile + 1, 2 * sub_group_tile + 3, tile / 2 + 1, tile / 3};
    sizes.erase(std::remove(sizes.begin(), sizes.end(), std::size_t(0)), sizes.end());
    if (OneWorkGroupOnly)
        sizes.erase(std::remove_if(sizes.begin(), sizes.end(), [tile](std::size_t s) { return s > tile; }),
                    sizes.end());
    std::sort(sizes.begin(), sizes.end());
    sizes.erase(std::unique(sizes.begin(), sizes.end()), sizes.end());
    return sizes;
}

int
main()
{
    bool run_test = false;
#if TEST_SYCL_RADIX_SORT_KT_AVAILABLE
    using Param = kt::kernel_param<TEST_DATA_PER_WORK_ITEM, TEST_WORK_GROUP_SIZE>;
    constexpr Param params;
    auto q = TestUtils::get_test_queue();
#    if TEST_KEYS_ONLY
    run_test = can_run_test<Param, KeyT>(q, params);
#    else
    run_test = can_run_test<Param, KeyT, ValueT>(q, params);
#    endif

    if (run_test)
    {
        try
        {
            const std::size_t tile = std::size_t(Param::data_per_workitem) * Param::workgroup_size;
            const DataMode modes[] = {DataMode::usm_iterators, DataMode::usm_ranges, DataMode::buffer_iterators,
                                      DataMode::buffer_ranges};
            std::size_t mode_idx = 0;
            for (std::size_t segment_size : segment_sizes(tile, std::size_t(Param::data_per_workitem) * 32))
            {
                // A single segment (segment_size == n), a few segments, and many segments. The count of the
                // smallest segments is capped: global scratch memory grows with the number of tiles and segments.
                const std::size_t many = std::clamp<std::size_t>((1 << 20) / segment_size, 2, 2048);
                for (std::size_t segment_count : {std::size_t(1), std::size_t(3), many})
                {
                    const DataMode mode = modes[mode_idx++ % 4];
                    test_case<Ascending, false>(q, mode, segment_size, segment_count,
                                                TestUtils::create_new_kernel_param_idx<0>(params));
                    test_case<Descending, false>(q, mode, segment_size, segment_count,
                                                 TestUtils::create_new_kernel_param_idx<1>(params));
                    test_case<Ascending, true>(q, mode, segment_size, segment_count,
                                               TestUtils::create_new_kernel_param_idx<2>(params));
                    test_case<Descending, true>(q, mode, segment_size, segment_count,
                                                TestUtils::create_new_kernel_param_idx<3>(params));
                }

                const DataMode mode = modes[mode_idx++ % 4];
                test_case<Ascending, false>(q, mode, segment_size, many,
                                            TestUtils::create_new_kernel_param_idx<0>(params), DataPattern::all_equal);
                test_case<Descending, true>(q, mode, segment_size, many,
                                            TestUtils::create_new_kernel_param_idx<3>(params), DataPattern::reversed);
#    if !TEST_KEYS_ONLY
                // Values record the position in the segment, so they must represent it exactly
                if (segment_size <= std::size_t(std::numeric_limits<ValueT>::max()))
                {
                    test_case<Ascending, false>(q, DataMode::usm_iterators, segment_size, 5,
                                                TestUtils::create_new_kernel_param_idx<0>(params),
                                                DataPattern::few_distinct);
                    test_case<Descending, true>(q, DataMode::buffer_ranges, segment_size, 5,
                                                TestUtils::create_new_kernel_param_idx<3>(params),
                                                DataPattern::few_distinct);
                    if constexpr (FloatingKey)
                    {
                        test_case<Ascending, true>(q, DataMode::usm_ranges, segment_size, 5,
                                                   TestUtils::create_new_kernel_param_idx<2>(params),
                                                   DataPattern::float_specials);
                        test_case<Descending, false>(q, DataMode::buffer_iterators, segment_size, 5,
                                                     TestUtils::create_new_kernel_param_idx<1>(params),
                                                     DataPattern::float_specials);
                    }
                }
#    else
                if constexpr (FloatingKey)
                {
                    test_case<Ascending, true>(q, DataMode::usm_ranges, segment_size, 5,
                                               TestUtils::create_new_kernel_param_idx<2>(params),
                                               DataPattern::float_specials);
                    test_case<Descending, false>(q, DataMode::buffer_iterators, segment_size, 5,
                                                 TestUtils::create_new_kernel_param_idx<1>(params),
                                                 DataPattern::float_specials);
                }
#    endif
            }

            // More histogram chunks than histogram work-groups, with segments of 3 chunks: work-groups own several
            // chunks and switch segments in the middle of their range
            if (!OneWorkGroupOnly || 9000 <= tile)
                test_case<Ascending, false>(q, DataMode::usm_iterators, 9000, 600,
                                            TestUtils::create_new_kernel_param_idx<0>(params));

            // n == 0 is a no-op
            KeyT* empty = nullptr;
            kt::gpu::batched_radix_sort<Ascending, BatchedRadixBits>(q, empty, empty, 4,
                                                                     TestUtils::create_new_kernel_param_idx<4>(params))
                .wait();
        }
        catch (const std::exception& exc)
        {
            std::cerr << "Exception: " << exc.what() << std::endl;
            return EXIT_FAILURE;
        }
    }
#endif // TEST_SYCL_RADIX_SORT_KT_AVAILABLE

    return TestUtils::done(run_test);
}
