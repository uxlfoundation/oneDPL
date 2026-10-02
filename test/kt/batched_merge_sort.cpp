// -*- C++ -*-
//===-- batched_merge_sort.cpp --------------------------------------------===//
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

namespace kt = oneapi::dpl::experimental::kt;

#ifndef TEST_VALUE_TYPE
#    define TEST_KEYS_ONLY 1
using ValueT = std::uint32_t; // unused by the sorts, only by the reference
#else
using ValueT = TEST_VALUE_TYPE;
#endif
using KeyT = TEST_KEY_TYPE;

constexpr bool Ascending = true;
constexpr bool Descending = false;

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
                e = kt::gpu::batched_merge_sort<IsAscending>(q, kp, kp + n, segment_size, param);
            else
                e = kt::gpu::batched_merge_sort<IsAscending>(q, kp, kp + n, kop, segment_size, param);
#else
            if constexpr (InPlace)
                e = kt::gpu::batched_merge_sort_by_key<IsAscending>(q, kp, kp + n, vp, segment_size, param);
            else
                e = kt::gpu::batched_merge_sort_by_key<IsAscending>(q, kp, kp + n, vp, kop, vop, segment_size, param);
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
                e = kt::gpu::batched_merge_sort<IsAscending>(q, kv, segment_size, param);
            else
                e = kt::gpu::batched_merge_sort<IsAscending>(q, kv, kov, segment_size, param);
#else
            if constexpr (InPlace)
                e = kt::gpu::batched_merge_sort_by_key<IsAscending>(q, kv, vv, segment_size, param);
            else
                e = kt::gpu::batched_merge_sort_by_key<IsAscending>(q, kv, vv, kov, vov, segment_size, param);
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
                    kt::gpu::batched_merge_sort<IsAscending>(q, kb, kb + n, segment_size, param).wait();
                else
                    kt::gpu::batched_merge_sort<IsAscending>(q, kb, kb + n, kob, segment_size, param).wait();
#else
                if constexpr (InPlace)
                    kt::gpu::batched_merge_sort_by_key<IsAscending>(q, kb, kb + n, vb, segment_size, param).wait();
                else
                    kt::gpu::batched_merge_sort_by_key<IsAscending>(q, kb, kb + n, vb, kob, vob, segment_size, param)
                        .wait();
#endif
            }
            else
            {
#if TEST_KEYS_ONLY
                if constexpr (InPlace)
                    kt::gpu::batched_merge_sort<IsAscending>(q, k, segment_size, param).wait();
                else
                    kt::gpu::batched_merge_sort<IsAscending>(q, k, ko, segment_size, param).wait();
#else
                if constexpr (InPlace)
                    kt::gpu::batched_merge_sort_by_key<IsAscending>(q, k, v, segment_size, param).wait();
                else
                    kt::gpu::batched_merge_sort_by_key<IsAscending>(q, k, v, ko, vo, segment_size, param).wait();
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
          bool stability_data = false)
{
    const std::size_t n = segment_size * segment_count;
    std::vector<KeyT> keys(n);
    std::vector<ValueT> vals(n);
    if (stability_data)
    {
        // Few distinct keys, values record the original position within the segment
        for (std::size_t i = 0; i < n; ++i)
        {
            keys[i] = KeyT((i * 7919) % 5);
            vals[i] = ValueT(i % segment_size);
        }
    }
    else
    {
        TestUtils::generate_arithmetic_data(keys.data(), n, 42 + segment_size);
        TestUtils::generate_arithmetic_data(vals.data(), n, 7);
    }

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
        << ", stability data: " << stability_data;
    const std::string m = msg.str();

    EXPECT_EQ_N(expected_keys.begin(), actual_keys.begin(), n, ("wrong keys, " + m).c_str());
#if !TEST_KEYS_ONLY
    EXPECT_EQ_N(expected_vals.begin(), actual_vals.begin(), n, ("wrong values, " + m).c_str());
#endif
    if constexpr (!InPlace)
    {
        EXPECT_EQ_N(keys.begin(), keys_in_after.begin(), n, ("input keys modified, " + m).c_str());
#if !TEST_KEYS_ONLY
        EXPECT_EQ_N(vals.begin(), vals_in_after.begin(), n, ("input values modified, " + m).c_str());
#endif
    }
}

std::vector<std::size_t>
segment_sizes(std::size_t dpwi, std::size_t capacity)
{
    std::vector<std::size_t> sizes = {1,       2,   3,   7,    dpwi,         dpwi + 1,         2 * dpwi - 1,
                                      100,     256, 317, 1000, capacity / 2, capacity / 2 + 3, capacity - 1,
                                      capacity};
    sizes.erase(std::remove_if(sizes.begin(), sizes.end(), [&](std::size_t s) { return s == 0 || s > capacity; }),
                sizes.end());
    std::sort(sizes.begin(), sizes.end());
    sizes.erase(std::unique(sizes.begin(), sizes.end()), sizes.end());
    return sizes;
}

template <typename KernelParam>
bool
can_run_test(sycl::queue q)
{
    std::size_t elem_bytes = sizeof(KeyT);
#if !TEST_KEYS_ONLY
    elem_bytes += sizeof(ValueT);
#endif
    const std::size_t slm = std::size_t(KernelParam::data_per_workitem) * KernelParam::workgroup_size * elem_bytes;
    const auto device = q.get_device();
    return slm <= device.get_info<sycl::info::device::local_mem_size>() &&
           KernelParam::workgroup_size <= device.get_info<sycl::info::device::max_work_group_size>();
}

int
main()
{
    using Param = kt::kernel_param<TEST_DATA_PER_WORK_ITEM, TEST_WORK_GROUP_SIZE>;
    constexpr Param params;
    auto q = TestUtils::get_test_queue();
    const bool run_test = can_run_test<Param>(q);

    if (run_test)
    {
        try
        {
            const std::size_t capacity = std::size_t(Param::data_per_workitem) * Param::workgroup_size;
            const DataMode modes[] = {DataMode::usm_iterators, DataMode::usm_ranges, DataMode::buffer_iterators,
                                      DataMode::buffer_ranges};
            std::size_t mode_idx = 0;
            for (std::size_t segment_size : segment_sizes(Param::data_per_workitem, capacity))
            {
                // Vary the segment count: a single segment, a partially filled last work-group, and many segments
                const std::size_t many = std::max<std::size_t>(3, (1 << 18) / segment_size);
                for (std::size_t segment_count : {std::size_t(1), std::size_t(37), many})
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
#if !TEST_KEYS_ONLY
                if (segment_size <= std::numeric_limits<ValueT>::max())
                {
                    test_case<Ascending, false>(q, DataMode::usm_iterators, segment_size, 5,
                                                TestUtils::create_new_kernel_param_idx<0>(params), true);
                    test_case<Descending, true>(q, DataMode::buffer_ranges, segment_size, 5,
                                                TestUtils::create_new_kernel_param_idx<3>(params), true);
                }
#endif
            }

            // n == 0 is a no-op
            KeyT* empty = nullptr;
            kt::gpu::batched_merge_sort(q, empty, empty, 4, TestUtils::create_new_kernel_param_idx<4>(params)).wait();
        }
        catch (const std::exception& exc)
        {
            std::cerr << "Exception: " << exc.what() << std::endl;
            return EXIT_FAILURE;
        }
    }

    return TestUtils::done(run_test);
}
