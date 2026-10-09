// -*- C++ -*-
//===-- batched_sort_test_utils.h -----------------------------------------===//
//
// Copyright (C) UXL Foundation Contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _BATCHED_SORT_TEST_UTILS_H
#define _BATCHED_SORT_TEST_UTILS_H

// Shared by the batched sort tests, which define KeyT, ValueT and TEST_KEYS_ONLY before including this header.
// A Sorter provides static sort<IsAscending>(q, args...) and sort_by_key<IsAscending>(q, args...), forwarding to the
// keys-only and key-value overloads of the sort under test.

#include <oneapi/dpl/experimental/kernel_templates>
#include <oneapi/dpl/iterator>

#include <algorithm>
#include <cstdint>
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

enum class DataMode
{
    usm_iterators,
    usm_ranges,
    buffer_iterators,
    buffer_ranges
};

inline const char*
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

inline const char*
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

inline std::vector<KeyBitsT>
key_bits(const std::vector<KeyT>& keys)
{
    std::vector<KeyBitsT> bits(keys.size());
    std::memcpy(bits.data(), keys.data(), keys.size() * sizeof(KeyT));
    return bits;
}

inline void
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

inline void
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

// Calls the in-place or out-of-place overload of the sort under test with the given keys, values and outputs: either
// iterators (first, last) or ranges
template <typename Sorter, bool IsAscending, bool InPlace, typename Keys, typename Vals, typename KeysOut,
          typename ValsOut, typename KernelParam>
sycl::event
invoke_sort([[maybe_unused]] sycl::queue q, [[maybe_unused]] Keys keys_first, [[maybe_unused]] Keys keys_last,
            [[maybe_unused]] Vals vals, [[maybe_unused]] KeysOut keys_out, [[maybe_unused]] ValsOut vals_out,
            [[maybe_unused]] std::size_t segment_size, [[maybe_unused]] KernelParam param, std::true_type /*iterators*/)
{
#if TEST_KEYS_ONLY
    if constexpr (InPlace)
        return Sorter::template sort<IsAscending>(q, keys_first, keys_last, segment_size, param);
    else
        return Sorter::template sort<IsAscending>(q, keys_first, keys_last, keys_out, segment_size, param);
#else
    if constexpr (InPlace)
        return Sorter::template sort_by_key<IsAscending>(q, keys_first, keys_last, vals, segment_size, param);
    else
        return Sorter::template sort_by_key<IsAscending>(q, keys_first, keys_last, vals, keys_out, vals_out,
                                                         segment_size, param);
#endif
}

template <typename Sorter, bool IsAscending, bool InPlace, typename Keys, typename Vals, typename KeysOut,
          typename ValsOut, typename KernelParam>
sycl::event
invoke_sort([[maybe_unused]] sycl::queue q, [[maybe_unused]] Keys&& keys, [[maybe_unused]] Vals&& vals,
            [[maybe_unused]] KeysOut&& keys_out, [[maybe_unused]] ValsOut&& vals_out,
            [[maybe_unused]] std::size_t segment_size, [[maybe_unused]] KernelParam param)
{
#if TEST_KEYS_ONLY
    if constexpr (InPlace)
        return Sorter::template sort<IsAscending>(q, keys, segment_size, param);
    else
        return Sorter::template sort<IsAscending>(q, keys, keys_out, segment_size, param);
#else
    if constexpr (InPlace)
        return Sorter::template sort_by_key<IsAscending>(q, keys, vals, segment_size, param);
    else
        return Sorter::template sort_by_key<IsAscending>(q, keys, vals, keys_out, vals_out, segment_size, param);
#endif
}

// Sorts on the device with the given data passing mode. The output vectors are both input and output for
// in-place sorts.
template <typename Sorter, bool IsAscending, bool InPlace, typename KernelParam>
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
            e = invoke_sort<Sorter, IsAscending, InPlace>(q, kp, kp + n, vp, kop, vop, segment_size, param,
                                                          std::true_type{});
        }
        else
        {
            // USM ranges are passed as subrange views over the pointers
            namespace views = oneapi::dpl::experimental::ranges::views;
            e = invoke_sort<Sorter, IsAscending, InPlace>(q, views::subrange(kp, kp + n), views::subrange(vp, vp + n),
                                                          views::subrange(kop, kop + n),
                                                          views::subrange(vop, vop + n), segment_size, param);
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
                invoke_sort<Sorter, IsAscending, InPlace>(q, kb, kb + n, oneapi::dpl::begin(v), oneapi::dpl::begin(ko),
                                                          oneapi::dpl::begin(vo), segment_size, param, std::true_type{})
                    .wait();
            }
            else
            {
                invoke_sort<Sorter, IsAscending, InPlace>(q, k, v, ko, vo, segment_size, param).wait();
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

template <typename Sorter, bool IsAscending, bool InPlace, typename KernelParam>
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
    device_sort<Sorter, IsAscending, InPlace>(q, mode, keys, vals, actual_keys, actual_vals, keys_in_after,
                                              vals_in_after, segment_size, param);

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

// Runs the patterns which only make sense with a few segments of the given size: few distinct keys (stability) and
// floating-point specials
template <typename Sorter, typename KernelParam>
void
test_special_patterns(sycl::queue q, std::size_t segment_size, KernelParam params)
{
#if !TEST_KEYS_ONLY
    // Values record the position in the segment, so they must represent it exactly
    if (segment_size > std::size_t(std::numeric_limits<ValueT>::max()))
        return;
    test_case<Sorter, true, false>(q, DataMode::usm_iterators, segment_size, 5,
                                   TestUtils::create_new_kernel_param_idx<0>(params), DataPattern::few_distinct);
    test_case<Sorter, false, true>(q, DataMode::buffer_ranges, segment_size, 5,
                                   TestUtils::create_new_kernel_param_idx<3>(params), DataPattern::few_distinct);
#endif
    if constexpr (FloatingKey)
    {
        test_case<Sorter, true, true>(q, DataMode::usm_ranges, segment_size, 5,
                                      TestUtils::create_new_kernel_param_idx<2>(params), DataPattern::float_specials);
        test_case<Sorter, false, false>(q, DataMode::buffer_iterators, segment_size, 5,
                                        TestUtils::create_new_kernel_param_idx<1>(params),
                                        DataPattern::float_specials);
    }
}

// n == 0 is a no-op for every overload. sycl::buffer cannot be empty, so only USM is covered.
template <typename Sorter, typename KernelParam>
void
test_empty_input(sycl::queue q, KernelParam params)
{
    namespace views = oneapi::dpl::experimental::ranges::views;
    KeyT* keys = nullptr;
    ValueT* vals = nullptr;
    invoke_sort<Sorter, true, true>(q, keys, keys, vals, keys, vals, 4,
                                    TestUtils::create_new_kernel_param_idx<4>(params), std::true_type{})
        .wait();
    invoke_sort<Sorter, true, false>(q, keys, keys, vals, keys, vals, 4,
                                     TestUtils::create_new_kernel_param_idx<5>(params), std::true_type{})
        .wait();
    invoke_sort<Sorter, true, true>(q, views::subrange(keys, keys), views::subrange(vals, vals),
                                    views::subrange(keys, keys), views::subrange(vals, vals), 4,
                                    TestUtils::create_new_kernel_param_idx<6>(params))
        .wait();
    invoke_sort<Sorter, true, false>(q, views::subrange(keys, keys), views::subrange(vals, vals),
                                     views::subrange(keys, keys), views::subrange(vals, vals), 4,
                                     TestUtils::create_new_kernel_param_idx<7>(params))
        .wait();
}

#endif // _BATCHED_SORT_TEST_UTILS_H
