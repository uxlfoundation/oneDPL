// -*- C++ -*-
//===-- parallel_backend_sycl_radix_sort.h --------------------------------===//
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

#ifndef _ONEDPL_PARALLEL_BACKEND_SYCL_RADIX_SORT_H
#define _ONEDPL_PARALLEL_BACKEND_SYCL_RADIX_SORT_H

#include <limits>
#include <type_traits>
#include <utility>
#include <cstdint>
#include <algorithm>
#include <functional> // for std::invoke

#include "sycl_defs.h"
#include "parallel_backend_sycl_utils.h"
#include "execution_sycl_defs.h"

#include "sycl_traits.h" //SYCL traits specialization for some oneDPL types.

namespace oneapi
{
namespace dpl
{
namespace __par_backend_hetero
{
//------------------------------------------------------------------------
// radix sort: bitwise order-preserving conversions to unsigned integrals
//------------------------------------------------------------------------

template <bool __is_ascending>
bool
__order_preserving_cast(bool __val)
{
    if constexpr (__is_ascending)
        return __val;
    else
        return !__val;
}

template <bool __is_ascending, typename _UInt, ::std::enable_if_t<::std::is_unsigned_v<_UInt>, int> = 0>
_UInt
__order_preserving_cast(_UInt __val)
{
    if constexpr (__is_ascending)
        return __val;
    else
        return ~__val; //bitwise inversion
}

template <bool __is_ascending, typename _Int,
          ::std::enable_if_t<::std::is_integral_v<_Int> && ::std::is_signed_v<_Int>, int> = 0>
::std::make_unsigned_t<_Int>
__order_preserving_cast(_Int __val)
{
    using _UInt = ::std::make_unsigned_t<_Int>;
    // mask: 100..0 for ascending, 011..1 for descending
    constexpr _UInt __mask =
        (__is_ascending) ? _UInt(1) << ::std::numeric_limits<_Int>::digits : ::std::numeric_limits<_UInt>::max() >> 1;
    return __val ^ __mask;
}

template <bool __is_ascending>
::std::uint16_t
__order_preserving_cast(sycl::half __val)
{
    ::std::uint16_t __uint16_val = oneapi::dpl::__internal::__dpl_bit_cast<::std::uint16_t>(__val);
    ::std::uint16_t __mask;
    // __uint16_val >> 15 takes the sign bit of the original value
    if constexpr (__is_ascending)
        __mask = (__uint16_val >> 15 == 0) ? 0x8000u : 0xFFFFu;
    else
        __mask = (__uint16_val >> 15 == 0) ? 0x7FFFu : ::std::uint16_t(0);
    return __uint16_val ^ __mask;
}

template <bool __is_ascending, typename _Float,
          ::std::enable_if_t<::std::is_floating_point_v<_Float> && sizeof(_Float) == sizeof(::std::uint32_t), int> = 0>
::std::uint32_t
__order_preserving_cast(_Float __val)
{
    ::std::uint32_t __uint32_val = oneapi::dpl::__internal::__dpl_bit_cast<::std::uint32_t>(__val);
    ::std::uint32_t __mask;
    // __uint32_val >> 31 takes the sign bit of the original value
    if constexpr (__is_ascending)
        __mask = (__uint32_val >> 31 == 0) ? 0x80000000u : 0xFFFFFFFFu;
    else
        __mask = (__uint32_val >> 31 == 0) ? 0x7FFFFFFFu : ::std::uint32_t(0);
    return __uint32_val ^ __mask;
}

template <bool __is_ascending, typename _Float,
          ::std::enable_if_t<::std::is_floating_point_v<_Float> && sizeof(_Float) == sizeof(::std::uint64_t), int> = 0>
::std::uint64_t
__order_preserving_cast(_Float __val)
{
    ::std::uint64_t __uint64_val = oneapi::dpl::__internal::__dpl_bit_cast<::std::uint64_t>(__val);
    ::std::uint64_t __mask;
    // __uint64_val >> 63 takes the sign bit of the original value
    if constexpr (__is_ascending)
        __mask = (__uint64_val >> 63 == 0) ? 0x8000000000000000u : 0xFFFFFFFFFFFFFFFFu;
    else
        __mask = (__uint64_val >> 63 == 0) ? 0x7FFFFFFFFFFFFFFFu : ::std::uint64_t(0);
    return __uint64_val ^ __mask;
}

//------------------------------------------------------------------------
// radix sort: bucket functions
//------------------------------------------------------------------------

// get number of buckets (size of radix bits) in T
template <typename _T>
constexpr ::std::uint32_t
__get_buckets_in_type(::std::uint32_t __radix_bits)
{
    return oneapi::dpl::__internal::__dpl_ceiling_div(sizeof(_T) * ::std::numeric_limits<unsigned char>::digits,
                                                      __radix_bits);
}

// get bits value (bucket) in a certain radix position
template <std::uint32_t __radix_mask, typename _T>
std::uint32_t
__get_bucket(_T __value, std::uint32_t __radix_offset)
{
    return (__value >> __radix_offset) & _T(__radix_mask);
}

//------------------------------------------------------------------------
// radix sort: kernel names
//------------------------------------------------------------------------

template <std::uint32_t, bool, typename... _Name>
class __radix_sort_count_kernel;

template <std::uint32_t, typename... _Name>
class __radix_sort_scan_kernel;

template <std::uint32_t, bool, typename... _Name>
class __radix_sort_reorder_kernel;

template <typename... _Name>
class __radix_sort_copy_back_kernel;

// Helper for SLM indexing in count kernel. Layout stores radix states contiguously per work-item
// for cache locality during counting, then reorganizes for reduction.
template <std::uint32_t __packing_ratio, std::uint32_t __radix_states>
struct __index_views
{
    // Number of radix state groups for the accumulation/reduction phase.
    // Must be >= __packing_ratio to ensure partial sums fit in the SLM layout.
    static constexpr std::uint32_t __counter_lanes = __radix_states / __packing_ratio;
    static constexpr std::uint32_t __num_groups =
        (__packing_ratio > __counter_lanes) ? __packing_ratio : __counter_lanes;
    // Number of radix states handled by each group
    static constexpr std::uint32_t __radix_states_per_group = __radix_states / __num_groups;

    // Index for uint8 bucket counters: [wg_id][radix_id]
    std::uint32_t
    __get_bucket_idx(std::uint32_t __radix_id, std::uint32_t __wg_id) const
    {
        return __wg_id * __radix_states + __radix_id;
    }

    // Index for packed uint32 counters (4 uint8s packed): [wg_id][radix_id_lane]
    std::uint32_t
    __get_bucket32_idx(std::uint32_t __radix_id_lane, std::uint32_t __wg_id) const
    {
        return __wg_id * (__radix_states / __packing_ratio) + __radix_id_lane;
    }

    // Index for reduction phase: [radix_id][partial_sum_id]
    // Stride per radix state = __wg_size / __num_groups
    std::uint32_t
    __get_count_idx(std::uint32_t __workgroup_size, std::uint32_t __radix_id, std::uint32_t __wg_id) const
    {
        return __radix_id * (__workgroup_size / __num_groups) + __wg_id;
    }
};

//-----------------------------------------------------------------------
// radix sort: count kernel helper
// Template helper function to count elements from a given input range.
// Called with a single runtime branch to select the input range.
//-----------------------------------------------------------------------
template <std::uint32_t __radix_bits, bool __is_ascending, typename _InputRange, typename _Proj, typename _IndexViews>
void
__radix_sort_count_impl(_InputRange& __input, _Proj __proj, std::uint32_t __radix_offset, std::size_t __sg_chunk_start,
                        std::size_t __sg_chunk_end, std::size_t __full_end, std::uint32_t __sg_size,
                        std::uint32_t __sg_local_id, std::size_t __wg_size, std::size_t __self_lidx,
                        std::uint8_t* __slm_buckets, _IndexViews __views)
{
    constexpr std::uint32_t __unroll_elements = 8;

    // Full iterations - no bounds checking needed
    for (std::size_t __base_idx = __sg_chunk_start + __sg_local_id; __base_idx < __full_end;
         __base_idx += __sg_size * __unroll_elements)
    {
        _ONEDPL_PRAGMA_UNROLL
        for (std::size_t __unroll = 0; __unroll < __unroll_elements; ++__unroll)
        {
            auto __val = __order_preserving_cast<__is_ascending>(
                std::invoke(__proj, __input[__base_idx + __unroll * __sg_size]));
            std::uint32_t __bucket = __get_bucket<(1 << __radix_bits) - 1>(__val, __radix_offset);
            ++__slm_buckets[__views.__get_bucket_idx(__bucket, __self_lidx)];
        }
    }

    // Remainder - at most one partial block per subgroup
    if (__full_end < __sg_chunk_end)
    {
        const std::size_t __base_idx = __full_end + __sg_local_id;
        _ONEDPL_PRAGMA_UNROLL
        for (std::size_t __unroll = 0; __unroll < __unroll_elements; ++__unroll)
        {
            std::size_t __curr_idx = __base_idx + __unroll * __sg_size;
            if (__curr_idx < __sg_chunk_end)
            {
                auto __val = __order_preserving_cast<__is_ascending>(std::invoke(__proj, __input[__curr_idx]));
                std::uint32_t __bucket = __get_bucket<(1 << __radix_bits) - 1>(__val, __radix_offset);
                ++__slm_buckets[__views.__get_bucket_idx(__bucket, __self_lidx)];
            }
        }
    }
}

//-----------------------------------------------------------------------
// radix sort: count kernel (per iteration)
//-----------------------------------------------------------------------

template <typename _KernelName, std::uint32_t __radix_bits, bool __is_ascending, std::uint32_t __unroll_elements = 8,
          typename _ValRange1, typename _ValRange2, typename _CountBuf, typename _Proj>
sycl::event
__radix_sort_count_submit(sycl::queue& __q, std::size_t __segments, std::size_t __wg_size,
                          std::uint32_t __radix_offset, bool __input_is_first, _ValRange1&& __val_rng1,
                          _ValRange2&& __val_rng2, _CountBuf& __count_buf, sycl::event __dependency_event, _Proj __proj)
{
    using _CountT = typename _CountBuf::value_type;

    // radix states used for an array storing bucket state counters
    constexpr std::uint32_t __radix_states = 1 << __radix_bits;
    // multiple uint8_t (1 byte) counters are packed in the footprint of 1 _CountT, we reuse SLM in different phases to
    // allow more parallel work to start with, still avoiding overflow as we accumulate bins from more elements together
    static constexpr std::uint32_t __packing_ratio = sizeof(_CountT);
    static constexpr std::uint32_t __counter_lanes = __radix_states / __packing_ratio;

    // iteration space info
    const std::size_t __n = oneapi::dpl::__ranges::__size(__val_rng1);
    const std::size_t __elem_per_segment = oneapi::dpl::__internal::__dpl_ceiling_div(__n, __segments);
    const std::size_t __no_op_flag_idx = __count_buf.size() - 1;

    //assert that we cannot overflow uint8 accumulation
    assert(oneapi::dpl::__internal::__dpl_ceiling_div(__elem_per_segment, __wg_size) < 256 &&
           "Segment size per work-group is too large to count in uint8");
    auto __count_rng =
        oneapi::dpl::__ranges::all_view<_CountT, __par_backend_hetero::access_mode::read_write>(__count_buf);

    // submit to compute arrays with local count values
    sycl::event __count_levent = __q.submit([&](sycl::handler& __hdl) {
        __hdl.depends_on(__dependency_event);

        // ensure the input data and the space for counters are accessible
        oneapi::dpl::__ranges::__require_access(__hdl, __val_rng1, __val_rng2, __count_rng);
        // an accessor per work-group with value counters from each work-item
        __dpl_sycl::__local_accessor<_CountT> __count_lacc(__radix_states * __wg_size / __packing_ratio, __hdl);
        __hdl.parallel_for<_KernelName>(
            sycl::nd_range<1>(__segments * __wg_size, __wg_size), [=](sycl::nd_item<1> __self_item) {
                // item info
                const std::size_t __self_lidx = __self_item.get_local_id(0);
                const std::size_t __wgroup_idx = __self_item.get_group(0);
                const std::size_t __seg_start = __elem_per_segment * __wgroup_idx;

                // Subgroup info for SG-strided memory access pattern
                auto __sub_group = __self_item.get_sub_group();
                const std::uint32_t __sg_size = __sub_group.get_local_range()[0];
                const std::uint32_t __sg_local_id = __sub_group.get_local_linear_id();
                const std::uint32_t __num_subgroups = oneapi::dpl::__internal::__dpl_ceiling_div(__wg_size, __sg_size);
                // Compute subgroup base from work item ID to handle variable subgroup sizes
                const std::size_t __sg_base = (__self_lidx - __sg_local_id);

                _CountT* __slm_counts = &__count_lacc[0];
                std::uint8_t* __slm_buckets = reinterpret_cast<std::uint8_t*>(__slm_counts);
                __index_views<__packing_ratio, __radix_states> __views;

                constexpr std::uint32_t __radix_states_per_group = decltype(__views)::__radix_states_per_group;
                _CountT __count_arr[__radix_states_per_group] = {0};
                const std::size_t __seg_end = sycl::min(__seg_start + __elem_per_segment, __n);

                // reset SLM buckets
                _ONEDPL_PRAGMA_UNROLL
                for (std::uint32_t __i = 0; __i < __counter_lanes; ++__i)
                {
                    __slm_counts[__views.__get_bucket32_idx(__i, __self_lidx)] = 0;
                }

                // SG-strided reads: each subgroup handles a contiguous chunk of the segment
                // Within each chunk, work items access memory with stride __sg_size for coalescing
                const std::size_t __seg_size = __seg_end - __seg_start;
                const std::size_t __elems_per_sg = __seg_size / __num_subgroups;
                const std::size_t __sg_id = __sg_base / __sg_size;
                const std::size_t __sg_chunk_start = __seg_start + __sg_id * __elems_per_sg;
                // Last subgroup gets any remainder elements
                const std::size_t __sg_chunk_end =
                    (__sg_id == __num_subgroups - 1) ? __seg_end : __sg_chunk_start + __elems_per_sg;

                const std::size_t __sg_chunk_size = __sg_chunk_end - __sg_chunk_start;
                const std::size_t __full_rounds = __sg_chunk_size / (__sg_size * __unroll_elements);
                const std::size_t __full_end = __sg_chunk_start + __full_rounds * __sg_size * __unroll_elements;

                // Single branch to select input range, then count without per-element branching
                if (__input_is_first)
                    __radix_sort_count_impl<__radix_bits, __is_ascending>(
                        __val_rng1, __proj, __radix_offset, __sg_chunk_start, __sg_chunk_end, __full_end, __sg_size,
                        __sg_local_id, __wg_size, __self_lidx, __slm_buckets, __views);
                else
                    __radix_sort_count_impl<__radix_bits, __is_ascending>(
                        __val_rng2, __proj, __radix_offset, __sg_chunk_start, __sg_chunk_end, __full_end, __sg_size,
                        __sg_local_id, __wg_size, __self_lidx, __slm_buckets, __views);

                __dpl_sycl::__group_barrier(__self_item);

                // Accumulation and reduction phase.
                // __num_groups work-item groups each handle __radix_states_per_group radix states.
                // Each WI reads __num_groups columns (covering all __wg_size WI counters).
                constexpr std::uint32_t __num_groups = decltype(__views)::__num_groups;
                const std::uint32_t __wis_per_radix_group = __wg_size / __num_groups;
                const std::uint32_t __radix_group = __self_lidx / __wis_per_radix_group;
                const std::uint32_t __radix_base = __radix_group * __radix_states_per_group;
                const std::uint32_t __wi_in_group = __self_lidx % __wis_per_radix_group;
                const std::uint32_t __col_base = __wi_in_group * __num_groups;

                // Accumulate __radix_states_per_group radix states from __num_groups columns
                _ONEDPL_PRAGMA_UNROLL
                for (std::uint32_t __c = 0; __c < __num_groups; ++__c)
                {
                    _ONEDPL_PRAGMA_UNROLL
                    for (std::uint32_t __r = 0; __r < __radix_states_per_group; ++__r)
                    {
                        __count_arr[__r] += static_cast<_CountT>(
                            __slm_buckets[__views.__get_bucket_idx(__radix_base + __r, __col_base + __c)]);
                    }
                }
                __dpl_sycl::__group_barrier(__self_item);

                // All WIs write their partial sums to SLM
                _ONEDPL_PRAGMA_UNROLL
                for (std::uint32_t __r = 0; __r < __radix_states_per_group; ++__r)
                {
                    __slm_counts[__views.__get_count_idx(__wg_size, __radix_base + __r, __wi_in_group)] =
                        __count_arr[__r];
                }
                __dpl_sycl::__group_barrier(__self_item);

                // Tree reduction: reduce partial sums down to 1 per radix state
                std::uint32_t __num_partial_sums = __wis_per_radix_group;
                for (std::uint32_t __stride = __num_partial_sums >> 1; __stride > 0; __stride >>= 1)
                {
                    // Each WI reduces its assigned radix states
                    if (__wi_in_group < __stride)
                    {
                        _ONEDPL_PRAGMA_UNROLL
                        for (std::uint32_t __r = 0; __r < __radix_states_per_group; ++__r)
                        {
                            __slm_counts[__views.__get_count_idx(__wg_size, __radix_base + __r, __wi_in_group)] +=
                                __slm_counts[__views.__get_count_idx(__wg_size, __radix_base + __r,
                                                                     __wi_in_group + __stride)];
                        }
                    }
                    __dpl_sycl::__group_barrier(__self_item);
                }

                // Write final count to global memory (only first 16 WIs, one per radix state)
                if (__self_lidx < __radix_states)
                {
                    __count_rng[(__segments + 1) * __self_lidx + __wgroup_idx] =
                        __slm_counts[__views.__get_count_idx(__wg_size, __self_lidx, 0)];
                }

                // Reset 'no operation flag' (indicates all keys are in one bin, skip reorder)
                if (__wgroup_idx == 0 && __self_lidx == 0)
                {
                    auto& __no_op_flag = __count_rng[__no_op_flag_idx];
                    __no_op_flag = 0;
                }
            });
    });

    return __count_levent;
}

//-----------------------------------------------------------------------
// radix sort: scan kernel (per iteration)
//-----------------------------------------------------------------------

template <typename _KernelName, std::uint32_t __radix_bits, typename _CountBuf>
sycl::event
__radix_sort_scan_submit(sycl::queue& __q, std::size_t __scan_wg_size, std::size_t __segments, _CountBuf& __count_buf,
                         std::size_t __n, sycl::event __dependency_event)
{
    using _CountT = typename _CountBuf::value_type;

    const std::size_t __no_op_flag_idx = __count_buf.size() - 1;
    auto __count_rng =
        oneapi::dpl::__ranges::all_view<_CountT, __par_backend_hetero::access_mode::read_write>(__count_buf);

    // Scan produces local offsets using count values.
    // There are no local offsets for the first segment, but the rest segments should be scanned
    // with respect to the count value in the first segment what requires n + 1 positions
    const std::size_t __scan_size = __segments + 1;
    __scan_wg_size = std::min(__scan_size, __scan_wg_size);

    const std::uint32_t __radix_states = 1 << __radix_bits;

    // compilation of the kernel prevents out of resources issue, which may occur due to usage of
    // collective algorithms such as joint_exclusive_scan even if local memory is not explicitly requested
    sycl::event __scan_event = __q.submit([&](sycl::handler& __hdl) {
        __hdl.depends_on(__dependency_event);
        // access the counters for all work groups
        oneapi::dpl::__ranges::__require_access(__hdl, __count_rng);
        __hdl.parallel_for<_KernelName>(
            sycl::nd_range<1>(__radix_states * __scan_wg_size, __scan_wg_size), [=](sycl::nd_item<1> __self_item) {
                // find borders of a region with a specific bucket id
                sycl::global_ptr<_CountT> __begin = __count_rng.begin() + __scan_size * __self_item.get_group(0);
                // TODO: consider another approach with use of local memory
                __dpl_sycl::__joint_exclusive_scan(__self_item.get_group(), __begin, __begin + __scan_size, __begin,
                                                   _CountT(0), __dpl_sycl::__plus<_CountT>{});
                const auto __wi = __self_item.get_local_linear_id();
                //That condition may be truth (by algo semantic) just on one WG, one WI, so there is no race here.
                if (__wi == __scan_wg_size - 1 && *(__begin + __scan_size - 1) == __n)
                {
                    auto& __no_op_flag = __count_rng[__no_op_flag_idx];
                    __no_op_flag = 1; //set flag if the all values got into one bin
                }
            });
    });
    return __scan_event;
}

//-----------------------------------------------------------------------
// radix sort: group level reorder algorithms
//-----------------------------------------------------------------------

template <typename _InputRange, typename _OutputRange>
void
__copy_kernel_for_radix_sort(sycl::nd_item<1> __self_item, const std::size_t __seg_start, std::size_t __seg_end,
                             const std::size_t __wg_size, _InputRange& __input, _OutputRange& __output)
{
    const std::size_t __self_lidx = __self_item.get_local_id(0);
    const std::uint16_t __residual = (__seg_end - __seg_start) % __wg_size;
    __seg_end -= __residual;

    for (std::size_t __val_idx = __seg_start + __self_lidx; __val_idx < __seg_end; __val_idx += __wg_size)
        __output[__val_idx] = std::move(__input[__val_idx]);

    if (__residual > 0 && __self_lidx < __residual)
    {
        const std::size_t __val_idx = __seg_end + __self_lidx;
        __output[__val_idx] = std::move(__input[__val_idx]);
    }
}

//-----------------------------------------------------------------------
// radix sort: reorder kernel helper
// Template helper function to reorder elements from input to output.
// Called with a single runtime branch to select the input/output ranges.
//-----------------------------------------------------------------------
template <std::uint32_t __radix_bits, bool __is_ascending, typename _InputRange, typename _OutputRange,
          typename _OffsetRange, typename _OffsetT, typename _ValT, typename _Proj>
void
__radix_sort_reorder_impl(_InputRange& __input, _OutputRange& __output, _OffsetRange& __offset_rng,
                          _OffsetT* __slm_counts, _ValT* __slm_vals, _OffsetT* __slm_global_offsets,
                          sycl::nd_item<1> __self_item, sycl::sub_group __sub_group,
                          _Proj __proj, std::uint32_t __radix_offset, std::size_t __segments, std::size_t __segment_idx,
                          std::size_t __seg_start, std::size_t __seg_end, std::uint32_t __sg_id,
                          std::uint32_t __sg_local_id, std::uint32_t __sg_size, std::uint32_t __num_subgroups)
{
    constexpr std::uint32_t __radix_states = 1 << __radix_bits;
    constexpr std::uint32_t __data_per_step = 8;
    const std::size_t __wg_size = __self_item.get_local_range(0);
    const std::uint32_t __wg_local_id = __self_item.get_local_linear_id();

    // Initialize global offsets in SLM
    if (__wg_local_id == 0)
    {
        _OffsetT __scanned_bin = 0;
        __slm_global_offsets[0] = __offset_rng[__segment_idx];
        const std::size_t __scan_size = __segments + 1;
        for (std::uint32_t __b = 1; __b < __radix_states; ++__b)
        {
            __scanned_bin += __offset_rng[__b * __scan_size - 1];
            __slm_global_offsets[__b] = __scanned_bin + __offset_rng[__segment_idx + __scan_size * __b];
        }
    }
    __dpl_sycl::__group_barrier(__self_item);

    _OffsetT* __step_bin_offset_SLM = __slm_counts + (__num_subgroups * __radix_states);

    for (std::size_t __step_start = __seg_start; __step_start < __seg_end; __step_start += __wg_size * __data_per_step)
    {
        const std::size_t __step_valid_count = sycl::min((std::size_t)(__wg_size * __data_per_step), __seg_end - __step_start);

        // 1. Load coalesced into SLM
        for (std::uint32_t __i = 0; __i < __data_per_step; ++__i)
        {
            std::size_t __idx = __wg_local_id + __i * __wg_size;
            if (__idx < __step_valid_count)
            {
                __slm_vals[__idx] = __input[__step_start + __idx];
            }
        }
        __dpl_sycl::__group_barrier(__self_item);

        // 2. Reorder to contiguous blocks in registers
        _ValT __regs[__data_per_step];
        std::uint32_t __bins[__data_per_step];
        _OffsetT __ranks[__data_per_step];

        // Initialize SLM hists for this step to 0
        for (std::uint32_t __b = __wg_local_id; __b < __num_subgroups * __radix_states; __b += __wg_size)
        {
            __slm_counts[__b] = 0;
        }
        __dpl_sycl::__group_barrier(__self_item);

        // 3. Process elements in registers and ballot-based scan
        // Stability fix: Process elements in subgroup stripes (0...31, then 32...63)
        const std::size_t __sg_step_base = __sg_id * (__sg_size * __data_per_step);

        for (std::uint32_t __i = 0; __i < __data_per_step; ++__i)
        {
            std::size_t __idx = __sg_step_base + __i * __sg_size + __sg_local_id;
            bool __is_valid_item = (__idx < __step_valid_count);
            if (__is_valid_item)
            {
                __regs[__i] = __slm_vals[__idx];
                auto __val = __order_preserving_cast<__is_ascending>(std::invoke(__proj, __regs[__i]));
                __bins[__i] = __get_bucket<(1 << __radix_bits) - 1>(__val, __radix_offset);
            }
            else
            {
                __bins[__i] = __radix_states; // Invalid bin
            }

            std::uint32_t __bin = __bins[__i];
            
            sycl::ext::oneapi::sub_group_mask __is_valid_lane = sycl::ext::oneapi::group_ballot(__sub_group, __bin < __radix_states);
            sycl::ext::oneapi::sub_group_mask __matched_mask = __is_valid_lane;
            _ONEDPL_PRAGMA_UNROLL
            for (int __bit_idx = 0; __bit_idx < __radix_bits; ++__bit_idx)
            {
                bool __bit = (__bin < __radix_states) && static_cast<bool>((__bin >> __bit_idx) & 1);
                sycl::ext::oneapi::sub_group_mask __sg_vote = sycl::ext::oneapi::group_ballot(__sub_group, __bit);
                
                if (__bin < __radix_states) {
                    if (__bit)
                        __matched_mask &= __sg_vote;
                    else
                        __matched_mask &= (~__sg_vote & __is_valid_lane);
                } else {
                    __matched_mask &= sycl::ext::oneapi::sub_group_mask{};
                }
            }

            // Sync before reading from SLM - ensures rank visibility across threads in the subgroup
            __dpl_sycl::__group_barrier(__self_item);
            _OffsetT __pre_rank = (__bin < __radix_states) ? __slm_counts[__sg_id * __radix_states + __bin] : 0;
            
            // Rank within subgroup
            uint64_t __matched_bits = 0;
            __matched_mask.extract_bits(__matched_bits);
            uint64_t __remove_right_lanes = (__sg_local_id >= 64) ? ~0ULL : (1ULL << __sg_local_id) - 1;
            uint32_t __this_round_rank = sycl::popcount(__matched_bits & __remove_right_lanes);
            uint32_t __this_round_count = sycl::popcount(__matched_bits);
            
            _OffsetT __rank_after = __pre_rank + __this_round_rank;
            bool __is_leader = (__bin < __radix_states && (__this_round_rank == __this_round_count - 1));
            
            // Sync before writing to SLM
            __dpl_sycl::__group_barrier(__self_item);
            if (__is_leader)
            {
                __slm_counts[__sg_id * __radix_states + __bin] = __rank_after + 1;
            }
            __ranks[__i] = __rank_after;
        }

        __dpl_sycl::__group_barrier(__self_item);

        // 4. Scan across histograms
        for (std::uint32_t __radix_state = __sg_id; __radix_state < __radix_states; __radix_state += __num_subgroups)
        {
            _OffsetT __running_sum = 0;
            for (std::uint32_t __base = 0; __base < __num_subgroups; __base += __sg_size)
            {
                const std::uint32_t __sg_idx = __base + __sg_local_id;
                _OffsetT __val = (__sg_idx < __num_subgroups) ? __slm_counts[__sg_idx * __radix_states + __radix_state] : 0;
                _OffsetT __local_prefix = __dpl_sycl::__exclusive_scan_over_group(__sub_group, __val, __dpl_sycl::__plus<_OffsetT>());
                _OffsetT __prefix = __running_sum + __local_prefix;
                
                if (__sg_idx < __num_subgroups)
                    __slm_counts[__sg_idx * __radix_states + __radix_state] = __prefix;
                    
                _OffsetT __chunk_total = __local_prefix + __val;
                __running_sum += __dpl_sycl::__group_broadcast(__sub_group, __chunk_total, __sg_size - 1);
            }
            if (__sg_local_id == 0)
            {
                __step_bin_offset_SLM[__radix_state] = __running_sum;
            }
        }

        __dpl_sycl::__group_barrier(__self_item);

        // Prefix sum of bin counts for this step
        if (__wg_local_id == 0)
        {
            _OffsetT __sum = 0;
            for (std::uint32_t __b = 0; __b < __radix_states; ++__b)
            {
                _OffsetT __count = __step_bin_offset_SLM[__b];
                __step_bin_offset_SLM[__b] = __sum;
                __sum += __count;
            }
        }
        __dpl_sycl::__group_barrier(__self_item);

        // 5. Write back to SLM grouping bins
        for (std::uint32_t __i = 0; __i < __data_per_step; ++__i)
        {
            if (__bins[__i] < __radix_states)
            {
                std::uint32_t __bin = __bins[__i];
                _OffsetT __slm_idx = __step_bin_offset_SLM[__bin] + 
                                     __slm_counts[__sg_id * __radix_states + __bin] + 
                                     __ranks[__i];
                __slm_vals[__slm_idx] = __regs[__i];
            }
        }

        __dpl_sycl::__group_barrier(__self_item);

        // 6. Write to back out to global memory in semi-coalesced way
        for (std::uint32_t __i = 0; __i < __data_per_step; ++__i)
        {
            std::size_t __slm_idx = __wg_local_id + __i * __wg_size;
            if (__slm_idx < __step_valid_count)
            {
                _ValT __val = __slm_vals[__slm_idx];
                
                auto __proj_val = __order_preserving_cast<__is_ascending>(std::invoke(__proj, __val));
                std::uint32_t __bin = __get_bucket<(1 << __radix_bits) - 1>(__proj_val, __radix_offset);
                
                _OffsetT __global_idx = __slm_global_offsets[__bin] + (__slm_idx - __step_bin_offset_SLM[__bin]);
                __output[__global_idx] = __val;
            }
        }

        __dpl_sycl::__group_barrier(__self_item);

        // 7. Update __slm_global_offsets for the next step
        if (__wg_local_id == 0)
        {
            for (std::uint32_t __b = 0; __b < __radix_states - 1; ++__b)
            {
                __slm_global_offsets[__b] += __step_bin_offset_SLM[__b + 1] - __step_bin_offset_SLM[__b];
            }
            __slm_global_offsets[__radix_states - 1] += __step_valid_count - __step_bin_offset_SLM[__radix_states - 1];
        }
        
        __dpl_sycl::__group_barrier(__self_item);
    }
}

//-----------------------------------------------------------------------
template <typename _KernelName, std::uint32_t __radix_bits, bool __is_ascending, typename _Range1, typename _Range2,
          typename _OffsetBuf, typename _Proj>
sycl::event
__radix_sort_reorder_submit(sycl::queue& __q, std::size_t __segments, std::size_t __wg_size, std::size_t __min_sg_size,
                            std::uint32_t __radix_offset, bool __input_is_first, _Range1&& __rng1, _Range2&& __rng2,
                            _OffsetBuf& __offset_buf, sycl::event __dependency_event, _Proj __proj)
{
    constexpr std::uint32_t __radix_states = 1 << __radix_bits;
    using _OffsetT = typename _OffsetBuf::value_type;

    assert(oneapi::dpl::__ranges::__size(__rng1) == oneapi::dpl::__ranges::__size(__rng2));

    const std::size_t __n = oneapi::dpl::__ranges::__size(__rng1);
    const std::size_t __elem_per_segment = oneapi::dpl::__internal::__dpl_ceiling_div(__n, __segments);
    const std::size_t __max_num_subgroups = __wg_size / __min_sg_size;
    const std::size_t __no_op_flag_idx = __offset_buf.size() - 1;

    auto __offset_rng =
        oneapi::dpl::__ranges::all_view<std::uint32_t, __par_backend_hetero::access_mode::read>(__offset_buf);

    sycl::event __reorder_event = __q.submit([&](sycl::handler& __hdl) {
        __hdl.depends_on(__dependency_event);
        oneapi::dpl::__ranges::__require_access(__hdl, __offset_rng);
        oneapi::dpl::__ranges::__require_access(__hdl, __rng1, __rng2);

        constexpr std::uint32_t __data_per_step = 8;
        auto __slm_counts = __dpl_sycl::__local_accessor<_OffsetT>(__max_num_subgroups * __radix_states + __radix_states, __hdl);
        auto __slm_global_offsets = __dpl_sycl::__local_accessor<_OffsetT>(__radix_states, __hdl);
        using _ValT = std::decay_t<decltype(__rng1[0])>;
        auto __slm_vals = __dpl_sycl::__local_accessor<_ValT>(__wg_size * __data_per_step, __hdl);

        __hdl.parallel_for<_KernelName>(
            sycl::nd_range<1>(__segments * __wg_size, __wg_size), [=](sycl::nd_item<1> __self_item) {
                const std::size_t __segment_idx = __self_item.get_group(0);
                const std::size_t __seg_start = __elem_per_segment * __segment_idx;
                const std::size_t __seg_end = sycl::min(__seg_start + __elem_per_segment, __n);

                auto& __no_op_flag = __offset_rng[__no_op_flag_idx];
                if (__no_op_flag)
                {
                    if (__input_is_first)
                        __copy_kernel_for_radix_sort(__self_item, __seg_start, __seg_end, __wg_size, __rng1, __rng2);
                    else
                        __copy_kernel_for_radix_sort(__self_item, __seg_start, __seg_end, __wg_size, __rng2, __rng1);
                    return;
                }

                auto __sub_group = __self_item.get_sub_group();
                const std::uint32_t __sg_id = __sub_group.get_group_linear_id();
                const std::uint32_t __sg_local_id = __sub_group.get_local_linear_id();
                const std::uint32_t __sg_size = __sub_group.get_local_range()[0];
                const std::uint32_t __num_subgroups = __wg_size / __sg_size;

                if (__input_is_first)
                    __radix_sort_reorder_impl<__radix_bits, __is_ascending>(
                        __rng1, __rng2, __offset_rng, &__slm_counts[0], &__slm_vals[0], &__slm_global_offsets[0],
                        __self_item, __sub_group, __proj, __radix_offset, __segments, __segment_idx,
                        __seg_start, __seg_end, __sg_id, __sg_local_id, __sg_size, __num_subgroups);
                else
                    __radix_sort_reorder_impl<__radix_bits, __is_ascending>(
                        __rng2, __rng1, __offset_rng, &__slm_counts[0], &__slm_vals[0], &__slm_global_offsets[0],
                        __self_item, __sub_group, __proj, __radix_offset, __segments, __segment_idx,
                        __seg_start, __seg_end, __sg_id, __sg_local_id, __sg_size, __num_subgroups);
            });
    });

    return __reorder_event;
}

//-----------------------------------------------------------------------

template <typename _KernelName, typename _Range1, typename _Range2>
sycl::event
__radix_sort_copy_back_submit(sycl::queue& __q, _Range1&& __in_rng, _Range2&& __out_rng, sycl::event __dependency_event)
{
    const std::size_t __n = oneapi::dpl::__ranges::__size(__in_rng);
    return __q.submit([&](sycl::handler& __hdl) {
        __hdl.depends_on(__dependency_event);
        oneapi::dpl::__ranges::__require_access(__hdl, __in_rng, __out_rng);
        __hdl.parallel_for<_KernelName>(sycl::range<1>(__n),
                                        [=](sycl::item<1> __item) { __in_rng[__item] = std::move(__out_rng[__item]); });
    });
}

//-----------------------------------------------------------------------
// radix sort: one iteration
//-----------------------------------------------------------------------

template <typename _CustomName, std::uint32_t __radix_bits, bool __is_ascending>
struct __parallel_multi_group_radix_sort
{
    template <typename... _Name>
    using __count_phase = __radix_sort_count_kernel<__radix_bits, __is_ascending, _Name...>;
    template <typename... _Name>
    using __local_scan_phase = __radix_sort_scan_kernel<__radix_bits, _Name...>;
    template <typename... _Name>
    using __reorder_phase = __radix_sort_reorder_kernel<__radix_bits, __is_ascending, _Name...>;
    template <typename... _Name>
    using __copy_back_phase = __radix_sort_copy_back_kernel<_Name...>;

    template <typename _InRange, typename _Proj>
    sycl::event
    operator()(sycl::queue& __q, _InRange&& __in_rng, _Proj __proj)
    {
        using _RadixCountKernel =
            __internal::__kernel_name_generator<__count_phase, _CustomName, std::decay_t<_InRange>, _Proj>;
        using _RadixLocalScanKernel = __internal::__kernel_name_generator<__local_scan_phase, _CustomName>;
        using _RadixReorderKernel =
            __internal::__kernel_name_generator<__reorder_phase, _CustomName, std::decay_t<_InRange>, _Proj>;
        using _RadixCopyBackKernel =
            __internal::__kernel_name_generator<__copy_back_phase, _CustomName, std::decay_t<_InRange>>;

        using _ValueT = oneapi::dpl::__internal::__value_t<_InRange>;
        using _KeyT = oneapi::dpl::__internal::__key_t<_Proj, _InRange>;

        constexpr std::uint32_t __radix_iters = __get_buckets_in_type<_KeyT>(__radix_bits);
        const std::uint32_t __radix_states = 1 << __radix_bits;
        const std::size_t __n = __in_rng.size();

        using _CounterType = std::uint32_t;
        std::size_t __wg_size_count = oneapi::dpl::__internal::__slm_adjusted_work_group_size(
            __q, sizeof(_CounterType) * __radix_states, std::size_t(128));
        // work-group size must be a power of 2 because of the tree reduction
        __wg_size_count =
            sycl::max(oneapi::dpl::__internal::__dpl_bit_floor(__wg_size_count), std::size_t(__radix_states));
        std::size_t __wg_size_scan = oneapi::dpl::__internal::__max_work_group_size(__q, 1024);
        std::size_t __wg_size_reorder = oneapi::dpl::__internal::__max_work_group_size(__q, 256);
        std::size_t __reorder_min_sg_size = oneapi::dpl::__internal::__min_sub_group_size(__q);

        constexpr std::uint32_t __unroll_elements = 8;
        // Keys per work-item in counting phase, recalculates based upon workgroup size for reorder phase.
        // We target a segment size of ~24KB to keep the scatter region in the reorder phase
        // within the L1 cache, preventing thrashing during uncoalesced memory accesses and scattered writes.
        constexpr std::size_t __target_segment_size_bytes = 24 * 1024;
        constexpr std::size_t __absolute_max_keys_per_wi = 255;
        constexpr std::size_t __absolute_min_keys_per_wi = __unroll_elements;
        constexpr std::size_t __target_minimum_segments = 128;

        static_assert(__absolute_max_keys_per_wi <= std::numeric_limits<unsigned char>::max(),
                      "Too large keys per work-item may cause overflow in counting phase");

        // target a number of segments
        std::size_t __keys_per_wi_count =
            oneapi::dpl::__internal::__dpl_ceiling_div(__n, (__wg_size_count * __target_minimum_segments));

        // cap total segment size
        __keys_per_wi_count =
            std::min(__keys_per_wi_count, __target_segment_size_bytes / (sizeof(_ValueT) * __wg_size_count));

        // apply hard limits
        __keys_per_wi_count = std::max(__keys_per_wi_count, __absolute_min_keys_per_wi);
        __keys_per_wi_count = std::min(__keys_per_wi_count, __absolute_max_keys_per_wi);

        const std::size_t __segments =
            oneapi::dpl::__internal::__dpl_ceiling_div(__n, __wg_size_count * __keys_per_wi_count);

        // Additional __radix_states elements are used for getting local offsets from count values + no_op flag;
        // 'No operation' flag specifies whether to skip re-order phase if the all keys are the same (lie in one bin)
        const std::size_t __tmp_buf_size = __segments * __radix_states + __radix_states + 1 /*no_op flag*/;
        // memory for storing count and offset values
        sycl::buffer<_CounterType, 1> __tmp_buf{sycl::range<1>(__tmp_buf_size)};

        // memory for storing values sorted for an iteration
        oneapi::dpl::__par_backend_hetero::__buffer<_ValueT> __out_buffer_holder{__n};
        auto __out_rng = oneapi::dpl::__ranges::all_view<_ValueT, __par_backend_hetero::access_mode::read_write>(
            __out_buffer_holder.get_buffer());

        // iterations per each bucket
        // TODO: radix for bool can be made using 1 iteration (x2 speedup against current implementation)
        sycl::event __dependency_event;
        for (std::uint32_t __radix_iter = 0; __radix_iter < __radix_iters; ++__radix_iter)
        {
            // TODO: convert to ordered type once at the first iteration and convert back at the last one
            bool __input_is_first = (__radix_iter % 2 == 0);
            // Compute the radix position for the given iteration
            std::uint32_t __radix_offset = __radix_iter * __radix_bits;

            // 1. Count Phase
            __dependency_event =
                __radix_sort_count_submit<_RadixCountKernel, __radix_bits, __is_ascending, __unroll_elements>(
                    __q, __segments, __wg_size_count, __radix_offset, __input_is_first, __in_rng, __out_rng, __tmp_buf,
                    __dependency_event, __proj);

            // 2. Scan Phase
            std::size_t __scan_size =
                __input_is_first ? oneapi::dpl::__ranges::__size(__in_rng) : oneapi::dpl::__ranges::__size(__out_rng);
            __dependency_event = __radix_sort_scan_submit<_RadixLocalScanKernel, __radix_bits>(
                __q, __wg_size_scan, __segments, __tmp_buf, __scan_size, __dependency_event);

            // 3. Reorder Phase
            __dependency_event = __radix_sort_reorder_submit<_RadixReorderKernel, __radix_bits, __is_ascending>(
                __q, __segments, __wg_size_reorder, __reorder_min_sg_size, __radix_offset, __input_is_first, __in_rng,
                __out_rng, __tmp_buf, __dependency_event, __proj);
        }

        // If odd number of iterations, the result is in __out_rng; copy back to __in_rng
        if constexpr (__radix_iters % 2 != 0)
        {
            __dependency_event =
                __radix_sort_copy_back_submit<_RadixCopyBackKernel>(__q, __in_rng, __out_rng, __dependency_event);
        }

        return __dependency_event;
    }
}; // struct __parallel_multi_group_radix_sort

// sorting by just one work group
#include "parallel_backend_sycl_radix_sort_one_wg.h"

//-----------------------------------------------------------------------
// radix sort: main function
//-----------------------------------------------------------------------
template <bool __is_ascending, typename _Range, typename _ExecutionPolicy, typename _Proj>
__future<sycl::event>
__parallel_radix_sort(oneapi::dpl::__internal::__device_backend_tag, _ExecutionPolicy&& __exec, _Range&& __in_rng,
                      _Proj __proj)
{
    const std::size_t __n = oneapi::dpl::__ranges::__size(__in_rng);
    assert(__n > 1);

    // radix bits represent number of processed bits in each value during one iteration
    constexpr std::uint32_t __radix_bits = 4;

    sycl::event __event;

    sycl::queue __q_local = __exec.queue();

    // Limit the work-group size to prevent large sizes on CPUs. Empirically found value.
    // This value exceeds the current practical limit for GPUs, but may need to be re-evaluated in the future.
    const std::size_t __max_wg_size = oneapi::dpl::__internal::__max_work_group_size(__q_local, (std::size_t)4096);
    const auto __subgroup_sizes = __q_local.get_device().template get_info<sycl::info::device::sub_group_sizes>();
    const bool __dev_has_sg16 = std::find(__subgroup_sizes.begin(), __subgroup_sizes.end(),
                                          static_cast<std::size_t>(16)) != __subgroup_sizes.end();

    using _RadixSortKernel = oneapi::dpl::__internal::__policy_kernel_name<_ExecutionPolicy>;

    // Select block size based on input size (block_size = elements per work-item)
    // Larger block sizes reduce register spills but require more registers per work-item
    if (__n <= std::min<std::size_t>(1024, __max_wg_size * 4))
        __event = __subgroup_radix_sort<_RadixSortKernel, 4, __radix_bits, __is_ascending>{}(
            __q_local, std::forward<_Range>(__in_rng), __proj, __max_wg_size);
    else if (__n <= std::min<std::size_t>(2048, __max_wg_size * 8))
        __event = __subgroup_radix_sort<_RadixSortKernel, 8, __radix_bits, __is_ascending>{}(
            __q_local, std::forward<_Range>(__in_rng), __proj, __max_wg_size);
    else if (__n <= std::min<std::size_t>(4096, __max_wg_size * 16))
        __event = __subgroup_radix_sort<_RadixSortKernel, 16, __radix_bits, __is_ascending>{}(
            __q_local, std::forward<_Range>(__in_rng), __proj, __max_wg_size);
    // In __subgroup_radix_sort, we request a sub-group size of 16 via _ONEDPL_SYCL_REQD_SUB_GROUP_SIZE_IF_SUPPORTED
    // for compilation targets that support this option. For the below cases, register spills that result in
    // runtime exceptions have been observed on accelerators that do not support the requested sub-group size of 16.
    // For the above cases that request but may not receive a sub-group size of 16, inputs are small enough to avoid
    // register spills on assessed hardware.
    else if (__n <= std::min<std::size_t>(16384, __max_wg_size * 32) && __dev_has_sg16)
        __event = __subgroup_radix_sort<_RadixSortKernel, 32, __radix_bits, __is_ascending>{}(
            __q_local, std::forward<_Range>(__in_rng), __proj, __max_wg_size);
    else
    {
        __event = __parallel_multi_group_radix_sort<_RadixSortKernel, __radix_bits, __is_ascending>{}(
            __q_local, std::forward<_Range>(__in_rng), __proj);
    }

    return __future{std::move(__event)};
}

} // namespace __par_backend_hetero
} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_PARALLEL_BACKEND_SYCL_RADIX_SORT_H
