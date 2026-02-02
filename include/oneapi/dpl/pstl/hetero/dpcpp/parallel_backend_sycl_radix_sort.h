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
#include "parallel_backend_sycl_reduce_then_scan.h" // for __sub_group_scan

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
template <::std::uint32_t __radix_mask, typename _T>
::std::uint32_t
__get_bucket(_T __value, ::std::uint32_t __radix_offset)
{
    return (__value >> __radix_offset) & _T(__radix_mask);
}

//------------------------------------------------------------------------
// radix sort: kernel names
//------------------------------------------------------------------------

template <::std::uint32_t, bool, typename... _Name>
class __radix_sort_count_kernel;

template <::std::uint32_t, typename... _Name>
class __radix_sort_scan_kernel;

template <::std::uint32_t, bool, typename... _Name>
class __radix_sort_reorder_kernel;

// Helper for SLM indexing in count kernel. Layout stores radix states contiguously per work-item
// for cache locality during counting, then reorganizes for reduction.
template <std::uint32_t __packing_ratio, std::uint32_t __radix_states>
struct __index_views
{
    // Index for uint8 bucket counters: [wg_id][radix_id]
    std::uint32_t
    `(std::uint32_t __radix_id, std::uint32_t __wg_id)
    {
        return __wg_id * __radix_states + __radix_id;
    }

    // Index for packed uint32 counters (4 uint8s packed): [wg_id][radix_id_lane]
    std::uint32_t
    __get_bucket32_idx(std::uint32_t __radix_id_lane, std::uint32_t __wg_id)
    {
        return __wg_id * (__radix_states / __packing_ratio) + __radix_id_lane;
    }

    // Index for reduction phase: [radix_id][partial_sum_id]
    std::uint32_t
    __get_count_idx(std::uint32_t __workgroup_size, std::uint32_t __radix_id, std::uint32_t __wg_id)
    {
        return __radix_id * (__workgroup_size / __packing_ratio) + __wg_id;
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
    {
        const std::size_t __base_idx = __full_end + __sg_local_id;
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
                          ::std::uint32_t __radix_offset, bool __input_is_first, _ValRange1&& __val_rng1,
                          _ValRange2&& __val_rng2, _CountBuf& __count_buf, sycl::event __dependency_event, _Proj __proj)
{
    // typedefs
    using _ValueT = oneapi::dpl::__internal::__value_t<_ValRange1>;
    using _CountT = typename _CountBuf::value_type;

    // radix states used for an array storing bucket state counters
    constexpr ::std::uint32_t __radix_states = 1 << __radix_bits;

    // iteration space info
    const ::std::size_t __n = oneapi::dpl::__ranges::__size(__val_rng1);
    const ::std::size_t __elem_per_segment = oneapi::dpl::__internal::__dpl_ceiling_div(__n, __segments);
    const ::std::size_t __no_op_flag_idx = __count_buf.size() - 1;

    //assert that we cannot overflow uint8 accumulation
    assert(__elem_per_segment / __wg_size < 256 && "Segment size per work-group is too large to count in uint8");

    auto __count_rng =
        oneapi::dpl::__ranges::all_view<_CountT, __par_backend_hetero::access_mode::read_write>(__count_buf);

    // submit to compute arrays with local count values
    sycl::event __count_levent = __q.submit([&](sycl::handler& __hdl) {
        __hdl.depends_on(__dependency_event);

        // ensure the input data and the space for counters are accessible
        oneapi::dpl::__ranges::__require_access(__hdl, __val_rng1, __val_rng2, __count_rng);
        // an accessor per work-group with value counters from each work-item
        auto __count_lacc = __dpl_sycl::__local_accessor<std::uint8_t>(__radix_states * __wg_size, __hdl);
        __hdl.parallel_for<_KernelName>(
            sycl::nd_range<1>(__segments * __wg_size, __wg_size), [=](sycl::nd_item<1> __self_item) {
                static constexpr std::uint32_t __packing_ratio = sizeof(_CountT) / sizeof(unsigned char);
                static constexpr std::uint32_t __counter_lanes = __radix_states / __packing_ratio;

                // item info
                const ::std::size_t __self_lidx = __self_item.get_local_id(0);
                const ::std::size_t __wgroup_idx = __self_item.get_group(0);
                const ::std::size_t __seg_start = __elem_per_segment * __wgroup_idx;

                // Subgroup info for SG-strided memory access pattern
                auto __sub_group = __self_item.get_sub_group();
                const std::uint32_t __sg_size = __sub_group.get_local_range()[0];
                const std::uint32_t __sg_local_id = __sub_group.get_local_linear_id();
                const std::uint32_t __num_subgroups = __wg_size / __sg_size;
                // Compute subgroup base from work item ID to handle variable subgroup sizes
                const ::std::size_t __sg_base = (__self_lidx - __sg_local_id);

                std::uint8_t* __slm_buckets = &__count_lacc[0];
                _CountT* __slm_counts = reinterpret_cast<_CountT*>(__slm_buckets);
                __index_views<__packing_ratio, __radix_states> __views;

                _CountT __count_arr[__packing_ratio] = {0};
                const ::std::size_t __seg_end = sycl::min(__seg_start + __elem_per_segment, __n);

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

                const std::uint32_t __wis_per_radix_group = __wg_size / __counter_lanes;
                const std::uint32_t __radix_group = __self_lidx / __wis_per_radix_group;
                const std::uint32_t __radix_base = __radix_group * __packing_ratio;
                const std::uint32_t __wi_in_group = __self_lidx % __wis_per_radix_group;
                const std::uint32_t __col_base = __wi_in_group * __packing_ratio;

                // Accumulate __packing_ratio radix states from __packing_ratio columns
                _ONEDPL_PRAGMA_UNROLL
                for (std::uint32_t __c = 0; __c < __packing_ratio; ++__c)
                {
                    _ONEDPL_PRAGMA_UNROLL
                    for (std::uint32_t __r = 0; __r < __packing_ratio; ++__r)
                    {
                        __count_arr[__r] += static_cast<_CountT>(
                            __slm_buckets[__views.__get_bucket_idx(__radix_base + __r, __col_base + __c)]);
                    }
                }
                __dpl_sycl::__group_barrier(__self_item);

                // All WIs write their __packing_ratio partial sums to SLM
                _ONEDPL_PRAGMA_UNROLL
                for (std::uint32_t __r = 0; __r < __packing_ratio; ++__r)
                {
                    __slm_counts[__views.__get_count_idx(__wg_size, __radix_base + __r, __wi_in_group)] =
                        __count_arr[__r];
                }
                __dpl_sycl::__group_barrier(__self_item);

                // Tree reduction: reduce 32 partial sums down to 1 per radix state
                // Layout after partial accumulation: radix_id * 32 + wi_in_group
                // Each WI is responsible for 4 radix states (__radix_base to __radix_base+3)
                std::uint32_t __num_partial_sums = __wg_size / __packing_ratio; // 32
                for (std::uint32_t __stride = __num_partial_sums >> 1; __stride > 0; __stride >>= 1)
                {
                    // Each WI reduces its assigned radix states
                    if (__wi_in_group < __stride)
                    {
                        _ONEDPL_PRAGMA_UNROLL
                        for (std::uint32_t __r = 0; __r < __packing_ratio; ++__r)
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
                         ::std::size_t __n, sycl::event __dependency_event)
{
    using _CountT = typename _CountBuf::value_type;

    const ::std::size_t __no_op_flag_idx = __count_buf.size() - 1;
    auto __count_rng =
        oneapi::dpl::__ranges::all_view<_CountT, __par_backend_hetero::access_mode::read_write>(__count_buf);

    // Scan produces local offsets using count values.
    // There are no local offsets for the first segment, but the rest segments should be scanned
    // with respect to the count value in the first segment what requires n + 1 positions
    const ::std::size_t __scan_size = __segments + 1;
    __scan_wg_size = ::std::min(__scan_size, __scan_wg_size);

    const ::std::uint32_t __radix_states = 1 << __radix_bits;

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
    const ::std::size_t __self_lidx = __self_item.get_local_id(0);
    const ::std::uint16_t __residual = (__seg_end - __seg_start) % __wg_size;
    __seg_end -= __residual;

    for (::std::size_t __val_idx = __seg_start + __self_lidx; __val_idx < __seg_end; __val_idx += __wg_size)
        __output[__val_idx] = std::move(__input[__val_idx]);

    if (__residual > 0 && __self_lidx < __residual)
    {
        const ::std::size_t __val_idx = __seg_end + __self_lidx;
        __output[__val_idx] = std::move(__input[__val_idx]);
    }
}

//-----------------------------------------------------------------------
// radix sort: reorder kernel helper
// Template helper function to reorder elements from input to output.
// Called with a single runtime branch to select the input/output ranges.
//-----------------------------------------------------------------------
template <std::uint32_t __radix_bits, bool __is_ascending, typename _InputRange, typename _OutputRange,
          typename _OffsetRange, typename _OffsetT, typename _Proj>
void
__radix_sort_reorder_impl(_InputRange& __input, _OutputRange& __output, _OffsetRange& __offset_rng,
                          _OffsetT* __slm_counts, sycl::nd_item<1> __self_item, sycl::sub_group __sub_group,
                          _Proj __proj, std::uint32_t __radix_offset, std::size_t __segments, std::size_t __segment_idx,
                          std::size_t __wi_start, std::size_t __wi_end, std::uint32_t __sg_id,
                          std::uint32_t __sg_local_id, std::uint32_t __sg_size, std::uint32_t __num_subgroups)
{
    constexpr std::uint32_t __radix_states = 1 << __radix_bits;

    // Phase 1: Count pass - each work-item counts its contiguous elements
    _OffsetT __local_counts[__radix_states] = {0};
    for (std::size_t __idx = __wi_start; __idx < __wi_end; ++__idx)
    {
        auto __val = __order_preserving_cast<__is_ascending>(std::invoke(__proj, __input[__idx]));
        ++__local_counts[__get_bucket<(1 << __radix_bits) - 1>(__val, __radix_offset)];
    }

    // Subgroup scan to get work-item prefix within subgroup
    // Last work-item writes totals directly to SLM (avoids broadcast)
    _OffsetT __wi_prefix[__radix_states];
    const bool __is_last_in_sg = (__sg_local_id == __sg_size - 1);
    for (std::uint32_t __b = 0; __b < __radix_states; ++__b)
    {
        __wi_prefix[__b] =
            __dpl_sycl::__exclusive_scan_over_group(__sub_group, __local_counts[__b], __dpl_sycl::__plus<_OffsetT>());
        if (__is_last_in_sg)
            __slm_counts[__sg_id * __radix_states + __b] = __wi_prefix[__b] + __local_counts[__b];
    }

    __dpl_sycl::__group_barrier(__self_item);

    // Phase 2: Compute subgroup prefix (subgroups loop through radix states)
    // Reuses the same SLM region: reads totals, computes prefix, writes back in-place
    for (std::uint32_t __radix_state = __sg_id; __radix_state < __radix_states; __radix_state += __num_subgroups)
    {
        _OffsetT __running_sum = 0;

        // Process counts in chunks of subgroup size
        for (std::uint32_t __base = 0; __base < __num_subgroups; __base += __sg_size)
        {
            const std::uint32_t __sg_idx = __base + __sg_local_id;

            // Load count (0 if out of bounds)
            _OffsetT __val = (__sg_idx < __num_subgroups) ? __slm_counts[__sg_idx * __radix_states + __radix_state] : 0;

            // Exclusive scan within chunk
            _OffsetT __local_prefix =
                __dpl_sycl::__exclusive_scan_over_group(__sub_group, __val, __dpl_sycl::__plus<_OffsetT>());

            // Add running sum from previous chunks
            _OffsetT __prefix = __running_sum + __local_prefix;

            // Write prefix back to same location (safe: all reads complete before any writes)
            if (__sg_idx < __num_subgroups)
                __slm_counts[__sg_idx * __radix_states + __radix_state] = __prefix;

            // Update running sum: broadcast the last element's total
            _OffsetT __chunk_total = __local_prefix + __val;
            __running_sum += __dpl_sycl::__group_broadcast(__sub_group, __chunk_total, __sg_size - 1);
        }
    }

    __dpl_sycl::__group_barrier(__self_item);

    // Phase 3: Compute final offsets = global_base + sg_prefix + wi_prefix
    _OffsetT __offsets[__radix_states];
    const std::size_t __scan_size = __segments + 1;
    _OffsetT __scanned_bin = 0;
    __offsets[0] = __offset_rng[__segment_idx] + __slm_counts[__sg_id * __radix_states] + __wi_prefix[0];

    for (std::uint32_t __b = 1; __b < __radix_states; ++__b)
    {
        __scanned_bin += __offset_rng[__b * __scan_size - 1];
        __offsets[__b] = __scanned_bin + __offset_rng[__segment_idx + __scan_size * __b] +
                         __slm_counts[__sg_id * __radix_states + __b] + __wi_prefix[__b];
    }

    // Phase 4: Scatter pass - re-read and write to output
    for (std::size_t __idx = __wi_start; __idx < __wi_end; ++__idx)
    {
        auto __in_val = __input[__idx];
        std::uint32_t __bucket = __get_bucket<(1 << __radix_bits) - 1>(
            __order_preserving_cast<__is_ascending>(std::invoke(__proj, __in_val)), __radix_offset);
        __output[__offsets[__bucket]++] = std::move(__in_val);
    }
}

//-----------------------------------------------------------------------
// radix sort: reorder kernel (per iteration)
//-----------------------------------------------------------------------
template <typename _KernelName, ::std::uint32_t __radix_bits, bool __is_ascending, typename _Range1, typename _Range2,
          typename _OffsetBuf, typename _Proj>
sycl::event
__radix_sort_reorder_submit(sycl::queue& __q, std::size_t __segments, std::size_t __wg_size, std::size_t __min_sg_size,
                            std::uint32_t __radix_offset, bool __input_is_first, _Range1&& __rng1, _Range2&& __rng2,
                            _OffsetBuf& __offset_buf, sycl::event __dependency_event, _Proj __proj)
{
    constexpr ::std::uint32_t __radix_states = 1 << __radix_bits;

    using _OffsetT = typename _OffsetBuf::value_type;

    assert(oneapi::dpl::__ranges::__size(__rng1) == oneapi::dpl::__ranges::__size(__rng2));

    // iteration space info
    const ::std::size_t __n = oneapi::dpl::__ranges::__size(__rng1);
    const ::std::size_t __elem_per_segment = oneapi::dpl::__internal::__dpl_ceiling_div(__n, __segments);
    const ::std::size_t __max_num_subgroups = __wg_size / __min_sg_size;

    const ::std::size_t __no_op_flag_idx = __offset_buf.size() - 1;

    auto __offset_rng =
        oneapi::dpl::__ranges::all_view<::std::uint32_t, __par_backend_hetero::access_mode::read>(__offset_buf);

    // submit to reorder values
    sycl::event __reorder_event = __q.submit([&](sycl::handler& __hdl) {
        __hdl.depends_on(__dependency_event);
        // access the offsets for all work groups
        oneapi::dpl::__ranges::__require_access(__hdl, __offset_rng);
        // access the input and output data
        oneapi::dpl::__ranges::__require_access(__hdl, __rng1, __rng2);

        // Minimal SLM: only for subgroup coordination (no value buffering)
        // Single region reused: first stores subgroup totals, then overwritten with prefix sums
        auto __slm_counts = __dpl_sycl::__local_accessor<_OffsetT>(__max_num_subgroups * __radix_states, __hdl);

        __hdl.parallel_for<_KernelName>(
            sycl::nd_range<1>(__segments * __wg_size, __wg_size), [=](sycl::nd_item<1> __self_item) {
                const std::size_t __segment_idx = __self_item.get_group(0);
                const std::size_t __seg_start = __elem_per_segment * __segment_idx;
                const std::size_t __seg_end = sycl::min(__seg_start + __elem_per_segment, __n);

                auto& __no_op_flag = __offset_rng[__no_op_flag_idx];
                if (__no_op_flag)
                {
                    // Single branch for copy: select input/output once, then copy without per-element branching
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

                // Compute this subgroup's contiguous chunk of the segment
                const std::size_t __elems_per_sg =
                    oneapi::dpl::__internal::__dpl_ceiling_div(__elem_per_segment, __num_subgroups);
                const std::size_t __sg_start = sycl::min(__seg_start + __sg_id * __elems_per_sg, __seg_end);
                const std::size_t __sg_end = sycl::min(__sg_start + __elems_per_sg, __seg_end);

                // Each work-item owns a contiguous block within its subgroup's chunk
                const std::size_t __sg_items = __sg_end - __sg_start;
                const std::size_t __items_per_wi = __sg_items / __sg_size;
                const std::size_t __wi_start = __sg_start + __sg_local_id * __items_per_wi;
                const std::size_t __wi_end =
                    (__sg_local_id == __sg_size - 1) ? __sg_end : (__wi_start + __items_per_wi);

                // Single branch to select input/output ranges, then reorder without per-element branching
                if (__input_is_first)
                    __radix_sort_reorder_impl<__radix_bits, __is_ascending>(
                        __rng1, __rng2, __offset_rng, &__slm_counts[0], __self_item, __sub_group, __proj,
                        __radix_offset, __segments, __segment_idx, __wi_start, __wi_end, __sg_id, __sg_local_id,
                        __sg_size, __num_subgroups);
                else
                    __radix_sort_reorder_impl<__radix_bits, __is_ascending>(
                        __rng2, __rng1, __offset_rng, &__slm_counts[0], __self_item, __sub_group, __proj,
                        __radix_offset, __segments, __segment_idx, __wi_start, __wi_end, __sg_id, __sg_local_id,
                        __sg_size, __num_subgroups);
            });
    });

    return __reorder_event;
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

    template <typename _InRange, typename _Proj>
    sycl::event
    operator()(sycl::queue& __q, _InRange&& __in_rng, _Proj __proj)
    {
        using _RadixCountKernel =
            __internal::__kernel_name_generator<__count_phase, _CustomName, std::decay_t<_InRange>, _Proj>;
        using _RadixLocalScanKernel = __internal::__kernel_name_generator<__local_scan_phase, _CustomName>;
        using _RadixReorderKernel =
            __internal::__kernel_name_generator<__reorder_phase, _CustomName, std::decay_t<_InRange>, _Proj>;

        using _ValueT = oneapi::dpl::__internal::__value_t<_InRange>;
        using _KeyT = oneapi::dpl::__internal::__key_t<_Proj, _InRange>;

        constexpr ::std::uint32_t __radix_iters = __get_buckets_in_type<_KeyT>(__radix_bits);
        const ::std::uint32_t __radix_states = 1 << __radix_bits;
        const ::std::size_t __n = __in_rng.size();

        using _CounterType = std::uint32_t;
        std::size_t __wg_size_count = oneapi::dpl::__internal::__slm_adjusted_work_group_size(
            __q, sizeof(_CounterType) * __radix_states, std::size_t(128));
        // work-group size must be a power of 2 because of the tree reduction
        __wg_size_count =
            sycl::max(oneapi::dpl::__internal::__dpl_bit_floor(__wg_size_count), ::std::size_t(__radix_states));
        std::size_t __wg_size_scan = oneapi::dpl::__internal::__max_work_group_size(__q, 1024);
        std::size_t __wg_size_reorder = oneapi::dpl::__internal::__max_work_group_size(__q, 256);
        std::size_t __reorder_min_sg_size = oneapi::dpl::__internal::__min_sub_group_size(__q);

        // Keys per work-item in counting phase, recalculates based upon workgroup size for reorder phase.
        // Empiracally found values, but here we check limits to prevent overflow in counting phase.
        constexpr std::size_t __keys_per_wi_count_max = 64;
        static_assert(__keys_per_wi_count_max < std::numeric_limits<unsigned char>::max(),
                      "Too large keys per work-item may cause overflow in counting phase");
        std::size_t __keys_per_wi_count = sycl::min(std::size_t(16), __keys_per_wi_count_max);
        if (__n >= 1 << 20)
        {
            __keys_per_wi_count = __keys_per_wi_count_max;
        }

        constexpr std::uint32_t __unroll_elements = 8;

        const ::std::size_t __segments =
            oneapi::dpl::__internal::__dpl_ceiling_div(__n, __wg_size_count * __keys_per_wi_count);

        // Additional __radix_states elements are used for getting local offsets from count values + no_op flag;
        // 'No operation' flag specifies whether to skip re-order phase if the all keys are the same (lie in one bin)
        const ::std::size_t __tmp_buf_size = __segments * __radix_states + __radix_states + 1 /*no_op flag*/;
        // memory for storing count and offset values
        sycl::buffer<_CounterType, 1> __tmp_buf{sycl::range<1>(__tmp_buf_size)};

        // memory for storing values sorted for an iteration
        oneapi::dpl::__par_backend_hetero::__buffer<_ValueT> __out_buffer_holder{__n};
        auto __out_rng = oneapi::dpl::__ranges::all_view<_ValueT, __par_backend_hetero::access_mode::read_write>(
            __out_buffer_holder.get_buffer());

        // iterations per each bucket
        assert("Number of iterations must be even" && __radix_iters % 2 == 0);
        // TODO: radix for bool can be made using 1 iteration (x2 speedup against current implementation)
        sycl::event __dependency_event;
        for (::std::uint32_t __radix_iter = 0; __radix_iter < __radix_iters; ++__radix_iter)
        {
            // TODO: convert to ordered type once at the first iteration and convert back at the last one
            bool __input_is_first = (__radix_iter % 2 == 0);
            // Compute the radix position for the given iteration
            ::std::uint32_t __radix_offset = __radix_iter * __radix_bits;

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
    const ::std::size_t __n = oneapi::dpl::__ranges::__size(__in_rng);
    assert(__n > 1);

    // radix bits represent number of processed bits in each value during one iteration
    constexpr ::std::uint32_t __radix_bits = 4;

    sycl::event __event;

    sycl::queue __q_local = __exec.queue();

    // Limit the work-group size to prevent large sizes on CPUs. Empirically found value.
    // This value exceeds the current practical limit for GPUs, but may need to be re-evaluated in the future.
    const std::size_t __max_wg_size = oneapi::dpl::__internal::__max_work_group_size(__q_local, (std::size_t)4096);

    //TODO: 1.to reduce number of the kernels; 2.to define work group size in runtime, depending on number of elements
    constexpr std::size_t __wg_size = 64;
    const auto __subgroup_sizes = __q_local.get_device().template get_info<sycl::info::device::sub_group_sizes>();
    const bool __dev_has_sg16 = std::find(__subgroup_sizes.begin(), __subgroup_sizes.end(),
                                          static_cast<std::size_t>(16)) != __subgroup_sizes.end();

    // _RadixSortKernel is used to generate unique kernel names for each instantiation of
    // __subgroup_radix_sort and __parallel_multi_group_radix_sort
    using _RadixSortKernel = oneapi::dpl::__internal::__policy_kernel_name<_ExecutionPolicy>;

    if (__n <= 64 && __wg_size <= __max_wg_size)
        __event = __subgroup_radix_sort<_RadixSortKernel, __wg_size, 1, __radix_bits, __is_ascending>{}(
            __q_local, std::forward<_Range>(__in_rng), __proj);
    else if (__n <= 128 && __wg_size * 2 <= __max_wg_size)
        __event = __subgroup_radix_sort<_RadixSortKernel, __wg_size * 2, 1, __radix_bits, __is_ascending>{}(
            __q_local, std::forward<_Range>(__in_rng), __proj);
    else if (__n <= 256 && __wg_size * 2 <= __max_wg_size)
        __event = __subgroup_radix_sort<_RadixSortKernel, __wg_size * 2, 2, __radix_bits, __is_ascending>{}(
            __q_local, std::forward<_Range>(__in_rng), __proj);
    else if (__n <= 512 && __wg_size * 2 <= __max_wg_size)
        __event = __subgroup_radix_sort<_RadixSortKernel, __wg_size * 2, 4, __radix_bits, __is_ascending>{}(
            __q_local, std::forward<_Range>(__in_rng), __proj);
    else if (__n <= 1024 && __wg_size * 2 <= __max_wg_size)
        __event = __subgroup_radix_sort<_RadixSortKernel, __wg_size * 2, 8, __radix_bits, __is_ascending>{}(
            __q_local, std::forward<_Range>(__in_rng), __proj);
    else if (__n <= 2048 && __wg_size * 4 <= __max_wg_size)
        __event = __subgroup_radix_sort<_RadixSortKernel, __wg_size * 4, 8, __radix_bits, __is_ascending>{}(
            __q_local, std::forward<_Range>(__in_rng), __proj);
    else if (__n <= 4096 && __wg_size * 4 <= __max_wg_size)
        __event = __subgroup_radix_sort<_RadixSortKernel, __wg_size * 4, 16, __radix_bits, __is_ascending>{}(
            __q_local, std::forward<_Range>(__in_rng), __proj);
    // In __subgroup_radix_sort, we request a sub-group size of 16 via _ONEDPL_SYCL_REQD_SUB_GROUP_SIZE_IF_SUPPORTED
    // for compilation targets that support this option. For the below cases, register spills that result in
    // runtime exceptions have been observed on accelerators that do not support the requested sub-group size of 16.
    // For the above cases that request but may not receive a sub-group size of 16, inputs are small enough to avoid
    // register spills on assessed hardware.
    else if (__n <= 8192 && __wg_size * 8 <= __max_wg_size && __dev_has_sg16)
        __event = __subgroup_radix_sort<_RadixSortKernel, __wg_size * 8, 16, __radix_bits, __is_ascending>{}(
            __q_local, std::forward<_Range>(__in_rng), __proj);
    else if (__n <= 16384 && __wg_size * 8 <= __max_wg_size && __dev_has_sg16)
        __event = __subgroup_radix_sort<_RadixSortKernel, __wg_size * 8, 32, __radix_bits, __is_ascending>{}(
            __q_local, std::forward<_Range>(__in_rng), __proj);
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
