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
// keys-only and key-value overloads of the sort under test. A Sorter with uses_temp_storage also provides
// bytes_req<KernelParam>(segment_size, n), and its sorts are given temporary storage of that many bytes.

#include <oneapi/dpl/experimental/kernel_templates>
#include <oneapi/dpl/iterator>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
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

// The generated keys contain no NaNs, so comparing with < or > matches the ordering of kt sorts (-0.0 == +0.0)
template <bool IsAscending>
void
reference_sort(std::vector<KeyT>& keys, std::vector<ValueT>& vals, std::size_t segment_size)
{
    auto zip_first = oneapi::dpl::make_zip_iterator(keys.begin(), vals.begin());
    for (std::size_t first = 0; first < keys.size(); first += segment_size)
        std::stable_sort(zip_first + first, zip_first + first + segment_size, CompareKey<IsAscending>{});
}

// Calls the in-place or out-of-place overload of the sort under test with the given keys, values and outputs: either
// iterators (first, last) or ranges. Extra arguments, such as temporary storage, precede the kernel_param.
template <typename Sorter, bool IsAscending, bool InPlace, typename Keys, typename Vals, typename KeysOut,
          typename ValsOut, typename KernelParam, typename... Extra>
sycl::event
invoke_sort(sycl::queue q, Keys keys_first, Keys keys_last, [[maybe_unused]] Vals vals, KeysOut keys_out,
            [[maybe_unused]] ValsOut vals_out, std::size_t segment_size, KernelParam param,
            std::true_type /*iterators*/, Extra... extra)
{
#if TEST_KEYS_ONLY
    if constexpr (InPlace)
        return Sorter::template sort<IsAscending>(q, keys_first, keys_last, segment_size, extra..., param);
    else
        return Sorter::template sort<IsAscending>(q, keys_first, keys_last, keys_out, segment_size, extra..., param);
#else
    if constexpr (InPlace)
        return Sorter::template sort_by_key<IsAscending>(q, keys_first, keys_last, vals, segment_size, extra..., param);
    else
        return Sorter::template sort_by_key<IsAscending>(q, keys_first, keys_last, vals, keys_out, vals_out,
                                                         segment_size, extra..., param);
#endif
}

template <typename Sorter, bool IsAscending, bool InPlace, typename Keys, typename Vals, typename KeysOut,
          typename ValsOut, typename KernelParam, typename... Extra>
sycl::event
invoke_sort(sycl::queue q, Keys&& keys, [[maybe_unused]] Vals&& vals, KeysOut&& keys_out,
            [[maybe_unused]] ValsOut&& vals_out, std::size_t segment_size, KernelParam param, Extra... extra)
{
#if TEST_KEYS_ONLY
    if constexpr (InPlace)
        return Sorter::template sort<IsAscending>(q, keys, segment_size, extra..., param);
    else
        return Sorter::template sort<IsAscending>(q, keys, keys_out, segment_size, extra..., param);
#else
    if constexpr (InPlace)
        return Sorter::template sort_by_key<IsAscending>(q, keys, vals, segment_size, extra..., param);
    else
        return Sorter::template sort_by_key<IsAscending>(q, keys, vals, keys_out, vals_out, segment_size, extra...,
                                                         param);
#endif
}

// Every distinct sort call needs a unique kernel name: the kernel templates only distinguish the kernels submitted
// within a single call
template <typename Sorter, bool IsAscending, bool InPlace, DataMode Mode, typename KernelParam>
constexpr auto
unique_kernel_param(KernelParam param)
{
    constexpr int variant = int(IsAscending) + 2 * int(InPlace) + 4 * int(Sorter::uses_temp_storage);
    return TestUtils::create_new_kernel_param_idx<4 * variant + int(Mode)>(param);
}

// Sorts on the device with the given data passing mode. The output vectors are both input and output for
// in-place sorts.
template <typename Sorter, bool IsAscending, bool InPlace, typename KernelParam, typename... Extra>
void
device_sort(sycl::queue q, DataMode mode, const std::vector<KeyT>& keys_in, const std::vector<ValueT>& vals_in,
            std::vector<KeyT>& keys_out, std::vector<ValueT>& vals_out, std::vector<KeyT>& keys_in_after,
            std::vector<ValueT>& vals_in_after, std::size_t segment_size, KernelParam param, Extra... extra)
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
            e = invoke_sort<Sorter, IsAscending, InPlace>(
                q, kp, kp + n, vp, kop, vop, segment_size,
                unique_kernel_param<Sorter, IsAscending, InPlace, DataMode::usm_iterators>(param), std::true_type{},
                extra...);
        }
        else
        {
            // USM ranges are passed as subrange views over the pointers
            namespace views = oneapi::dpl::experimental::ranges::views;
            e = invoke_sort<Sorter, IsAscending, InPlace>(
                q, views::subrange(kp, kp + n), views::subrange(vp, vp + n), views::subrange(kop, kop + n),
                views::subrange(vop, vop + n), segment_size,
                unique_kernel_param<Sorter, IsAscending, InPlace, DataMode::usm_ranges>(param), extra...);
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
                invoke_sort<Sorter, IsAscending, InPlace>(
                    q, kb, kb + n, oneapi::dpl::begin(v), oneapi::dpl::begin(ko), oneapi::dpl::begin(vo), segment_size,
                    unique_kernel_param<Sorter, IsAscending, InPlace, DataMode::buffer_iterators>(param),
                    std::true_type{}, extra...)
                    .wait();
            }
            else
            {
                invoke_sort<Sorter, IsAscending, InPlace>(
                    q, k, v, ko, vo, segment_size,
                    unique_kernel_param<Sorter, IsAscending, InPlace, DataMode::buffer_ranges>(param), extra...)
                    .wait();
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
    if constexpr (Sorter::uses_temp_storage)
    {
        // Exactly the reported number of bytes, at an address only aligned to a byte
        const std::size_t bytes = Sorter::template bytes_req<KernelParam>(segment_size, n);
        TestUtils::usm_data_transfer<sycl::usm::alloc::device, std::uint8_t> mem(q, bytes + 1);
        device_sort<Sorter, IsAscending, InPlace>(q, mode, keys, vals, actual_keys, actual_vals, keys_in_after,
                                                  vals_in_after, segment_size, param, mem.get_data() + 1, bytes);
    }
    else
    {
        device_sort<Sorter, IsAscending, InPlace>(q, mode, keys, vals, actual_keys, actual_vals, keys_in_after,
                                                  vals_in_after, segment_size, param);
    }

    std::ostringstream msg;
    msg << mode_name(mode) << ", temp storage: " << Sorter::uses_temp_storage << ", in-place: " << InPlace
        << ", ascending: " << IsAscending << ", segment size: " << segment_size << ", segments: " << segment_count
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
    // Values record positions in the segment and must represent them exactly. Conservatively cap the segment
    // size at 2^digits - 1 for both integer and binary floating-point types.
    // If ValueT has at least as many precision bits as size_t, no cap is needed and the shift is avoided.
    constexpr int digits = std::numeric_limits<ValueT>::digits;
    if constexpr (digits < std::numeric_limits<std::size_t>::digits)
    {
        if (segment_size > (std::size_t{1} << digits) - 1)
            return;
    }
    test_case<Sorter, true, false>(q, DataMode::usm_iterators, segment_size, 5, params, DataPattern::few_distinct);
    test_case<Sorter, false, true>(q, DataMode::buffer_ranges, segment_size, 5, params, DataPattern::few_distinct);
#endif
    if constexpr (FloatingKey)
    {
        test_case<Sorter, true, true>(q, DataMode::usm_ranges, segment_size, 5, params, DataPattern::float_specials);
        test_case<Sorter, false, false>(q, DataMode::buffer_iterators, segment_size, 5, params,
                                        DataPattern::float_specials);
    }
}

// n == 0 is a no-op for every overload, and needs no temporary storage. sycl::buffer cannot be empty, so only USM is
// covered.
template <typename Sorter, typename KernelParam, typename... Extra>
void
test_empty_input(sycl::queue q, KernelParam params, Extra... extra)
{
    namespace views = oneapi::dpl::experimental::ranges::views;
    KeyT* keys = nullptr;
    ValueT* vals = nullptr;
    invoke_sort<Sorter, true, true>(q, keys, keys, vals, keys, vals, 4,
                                    unique_kernel_param<Sorter, true, true, DataMode::usm_iterators>(params),
                                    std::true_type{}, extra...)
        .wait();
    invoke_sort<Sorter, true, false>(q, keys, keys, vals, keys, vals, 4,
                                     unique_kernel_param<Sorter, true, false, DataMode::usm_iterators>(params),
                                     std::true_type{}, extra...)
        .wait();
    invoke_sort<Sorter, true, true>(q, views::subrange(keys, keys), views::subrange(vals, vals),
                                    views::subrange(keys, keys), views::subrange(vals, vals), 4,
                                    unique_kernel_param<Sorter, true, true, DataMode::usm_ranges>(params), extra...)
        .wait();
    invoke_sort<Sorter, true, false>(q, views::subrange(keys, keys), views::subrange(vals, vals),
                                     views::subrange(keys, keys), views::subrange(vals, vals), 4,
                                     unique_kernel_param<Sorter, true, false, DataMode::usm_ranges>(params), extra...)
        .wait();
}

#endif // _BATCHED_SORT_TEST_UTILS_H
