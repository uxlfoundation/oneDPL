// -*- C++ -*-
//===-- sycl_radix_sort_kernels.h ----------------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _ONEDPL_KT_SYCL_RADIX_SORT_KERNELS_H
#define _ONEDPL_KT_SYCL_RADIX_SORT_KERNELS_H

#include <cstdint>
#include <type_traits>

#include "../../../pstl/hetero/dpcpp/sycl_defs.h"
#include "../../../pstl/utils.h"
#include "../../../pstl/hetero/dpcpp/utils_ranges_sycl.h"

#include "sub_group/sub_group_scan.h"

#include "radix_sort_kernels.h"
#include "radix_sort_utils.h"

namespace oneapi::dpl::experimental::kt::gpu::__impl
{

//-----------------------------------------------------------------------------
// SYCL global histogram kernel implementation
//-----------------------------------------------------------------------------
template <bool __is_ascending, std::uint8_t __radix_bits, std::uint32_t __hist_work_group_count,
          std::uint16_t __hist_work_group_size, typename _KeysRng>
struct __global_histogram<__sycl_tag, __is_ascending, __radix_bits, __hist_work_group_count, __hist_work_group_size,
                          _KeysRng>
{
    using _KeyT = oneapi::dpl::__internal::__value_t<_KeysRng>;
    using _BinT = std::uint16_t;
    using _GlobOffsetT = std::uint32_t;
    using _LocIdxT = std::uint32_t;

    static constexpr std::uint32_t __sub_group_size = 32;
    static constexpr std::uint32_t __hist_num_sub_groups = __hist_work_group_size / __sub_group_size;
    static constexpr std::uint32_t __bin_count = 1 << __radix_bits;
    static constexpr std::uint32_t __bit_count = sizeof(_KeyT) * 8;
    static constexpr std::uint32_t __stage_count =
        oneapi::dpl::__internal::__dpl_ceiling_div(__bit_count, __radix_bits);
    static constexpr std::uint32_t __hist_data_per_sub_group = 128;
    static constexpr std::uint32_t __hist_data_per_work_item = __hist_data_per_sub_group / __sub_group_size;
    static constexpr std::uint32_t __device_wide_step =
        __hist_work_group_count * __hist_work_group_size * __hist_data_per_work_item;
    static constexpr std::uint32_t __hist_buffer_size = __stage_count * __bin_count;

    std::size_t __n;
    _KeysRng __keys_rng;
    sycl::local_accessor<std::uint32_t, 1> __slm_acc;
    std::uint32_t* __p_global_offset;
    std::uint32_t __num_histograms;

    __global_histogram(std::size_t __n, const _KeysRng& __keys_rng, sycl::local_accessor<std::uint32_t, 1> __slm_acc,
                       std::uint32_t* __p_global_offset, std::uint32_t __num_histograms)
        : __n(__n), __keys_rng(__keys_rng), __slm_acc(__slm_acc), __p_global_offset(__p_global_offset),
          __num_histograms(__num_histograms)
    {
    }

    [[sycl::reqd_sub_group_size(__sub_group_size)]] void
    operator()(sycl::nd_item<1> __idx) const
    {
        std::uint32_t* __slm = __slm_acc.get_multi_ptr<sycl::access::decorated::no>().get();

        const std::uint32_t __local_id = __idx.get_local_linear_id();
        const std::uint32_t __group_id = __idx.get_group_linear_id();
        const std::uint32_t __sub_group_id = __idx.get_sub_group().get_group_linear_id();
        const std::uint32_t __sub_group_local_id = __idx.get_sub_group().get_local_linear_id();

        _GlobOffsetT __sub_group_start =
            (__group_id * __hist_num_sub_groups + __sub_group_id) * __hist_data_per_sub_group;

        // 0. Early exit - important for small inputs as we intentionally oversubscribe the hardware
        if ((__sub_group_start - __sub_group_id * __hist_data_per_sub_group) >= __n)
            return;

        // 1. Initialize group-local histograms in SLM
        for (_LocIdxT __i = __local_id; __i < __hist_buffer_size; __i += __hist_work_group_size)
        {
            _ONEDPL_PRAGMA_UNROLL
            for (_LocIdxT __j = 0; __j < __num_histograms; ++__j)
            {
                __slm[__i * __num_histograms + __j] = 0;
            }
        }

        __dpl_sycl::__group_barrier(__idx);

        for (_GlobOffsetT __wi_offset = __sub_group_start + __sub_group_local_id; __wi_offset < __n;
             __wi_offset += __device_wide_step)
        {
            // Keys loaded with stride of sub-group size
            _KeyT __keys[__hist_data_per_work_item];

            // 2. Read __keys
            if (__wi_offset + __hist_data_per_sub_group <= __n)
            {
                _ONEDPL_PRAGMA_UNROLL
                for (std::uint32_t __i = 0; __i != __hist_data_per_work_item; ++__i)
                {
                    __keys[__i] = __keys_rng[__i * __sub_group_size + __wi_offset];
                }
            }
            else
            {
                for (std::uint32_t __i = 0; __i != __hist_data_per_work_item; ++__i)
                {
                    std::size_t __key_idx = __i * __sub_group_size + __wi_offset;
                    __keys[__i] = (__key_idx < __n) ? __keys_rng[__key_idx] : __sort_identity<_KeyT, __is_ascending>();
                }
            }

            // 3. Accumulate histogram to SLM
            // SLM uses a blocked layout where each bin contains _NumHistograms sub-bins that are used to reduce
            // contention during atomic accumulation.
            // Use sub group local id to randomize sub-bin selection for histogram accumulation
            _LocIdxT __slm_hist_lane_offset = __sub_group_local_id % __num_histograms;
            _ONEDPL_PRAGMA_UNROLL
            for (std::uint32_t __stage = 0; __stage < __stage_count; ++__stage)
            {
                constexpr _BinT __mask = __bin_count - 1;
                _ONEDPL_PRAGMA_UNROLL
                for (std::uint32_t __i = 0; __i < __hist_data_per_work_item; ++__i)
                {
                    _BinT __bucket = __get_bucket_scalar<__mask>(
                        __order_preserving_cast_scalar<__is_ascending>(__keys[__i]), __stage * __radix_bits);
                    _GlobOffsetT __bin = __stage * __bin_count + __bucket;
                    using _SLMAtomicRef =
                        sycl::atomic_ref<_GlobOffsetT, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                         sycl::access::address_space::local_space>;
                    _SLMAtomicRef __slm_ref(__slm[__bin * __num_histograms + __slm_hist_lane_offset]);
                    __slm_ref.fetch_add(1);
                }
            }
        }

        __dpl_sycl::__group_barrier(__idx);

        // 4. Reduce group-local histograms from SLM into global histograms in global memory
        for (_LocIdxT __i = __local_id; __i < __hist_buffer_size; __i += __hist_work_group_size)
        {
            using _AtomicRef = sycl::atomic_ref<_GlobOffsetT, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                                sycl::access::address_space::global_space>;
            _GlobOffsetT __reduced_bincount = 0;
            // Blocked layout enables load vectorization from SLM
            _ONEDPL_PRAGMA_UNROLL
            for (_LocIdxT __j = 0; __j < __num_histograms; ++__j)
            {
                __reduced_bincount += __slm[__i * __num_histograms + __j];
            }
            _AtomicRef __global_hist_ref(__p_global_offset[__i]);
            __global_hist_ref.fetch_add(__reduced_bincount);
        }
    }
};

template <typename _KtTag, bool __is_ascending, std::uint8_t __radix_bits, std::uint16_t __data_per_work_item,
          std::uint16_t __work_group_size, typename _InRngPack, typename _OutRngPack>
struct __radix_sort_onesweep_kernel;

template <bool __is_ascending, std::uint8_t __radix_bits, std::uint16_t __data_per_work_item,
          std::uint16_t __work_group_size, typename _InRngPack, typename _OutRngPack>
struct __radix_sort_onesweep_kernel<__sycl_tag, __is_ascending, __radix_bits, __data_per_work_item, __work_group_size,
                                    _InRngPack, _OutRngPack>
{
    using _LocOffsetT = std::uint16_t;
    using _GlobOffsetT = std::uint32_t;
    using _AtomicIdT = std::uint32_t;
    using _LocIdxT = std::uint32_t;

    using _SubGroupBitmaskT = std::uint32_t;

    using _KeyT = typename _InRngPack::_KeyT;
    using _ValT = typename _InRngPack::_ValT;
    static constexpr bool __has_values = !std::is_void_v<_ValT>;

    static constexpr std::uint32_t __bin_count = 1 << __radix_bits;

    static constexpr std::uint32_t __sub_group_size = 32;
    static constexpr std::uint32_t __num_sub_groups_per_work_group = __work_group_size / __sub_group_size;
    static constexpr std::uint32_t __data_per_sub_group = __data_per_work_item * __sub_group_size;

    static constexpr std::uint32_t __bit_count = sizeof(_KeyT) * 8;
    static constexpr _LocOffsetT __mask = __bin_count - 1;
    static constexpr std::uint32_t __hist_stride = __bin_count * sizeof(_LocOffsetT);
    static constexpr std::uint32_t __work_item_all_hists_size = __num_sub_groups_per_work_group * __hist_stride;
    static constexpr std::uint32_t __group_hist_size = __hist_stride; // _LocOffsetT
    static constexpr std::uint32_t __global_hist_size = __bin_count * sizeof(_GlobOffsetT);

    static constexpr std::uint32_t
    __calc_reorder_slm_size()
    {
        if constexpr (__has_values)
            return __work_group_size * __data_per_work_item * (sizeof(_KeyT) + sizeof(_ValT));
        else
            return __work_group_size * __data_per_work_item * sizeof(_KeyT);
    }

    static constexpr std::uint32_t
    __get_slm_group_hist_offset()
    {
        constexpr std::uint32_t __reorder_size = __calc_reorder_slm_size();
        return std::max(__work_item_all_hists_size, __reorder_size);
    }

    static constexpr std::uint32_t
    __get_slm_global_incoming_offset()
    {
        return __get_slm_group_hist_offset() + __group_hist_size;
    }

    static constexpr std::uint32_t
    __calc_slm_alloc()
    {
        // SLM Layout Visualization:
        //
        // Phase 1 (Offset Calculation):
        // ┌──────────────────────────┬──────────────┬──────────────────┐
        // │   Sub-group Hists        │  Group Hist  │ Global Incoming  │
        // │ max(__work_item_all_     │  __group_    │ __global_hist    │
        // │     hists_size,          │  hist_size   │     _size        │
        // │     __reorder_size)      │              │                  │
        // └──────────────────────────┴──────────────┴──────────────────┘
        //                                    │              │
        //                                    v              v
        // Phase 2 (Reorder):
        // ┌──────────────────────────┬──────────────┬──────────────────┐
        // │   Reorder Space          │  Group Hist  │   Global Fix     │
        // │ max(__work_item_all_     │  __group_    │  __global_hist   │
        // │     hists_size,          │  hist_size   │      _size       │
        // │     __reorder_size)      │              │                  │
        // └──────────────────────────┴──────────────┴──────────────────┘
        //
        constexpr std::uint32_t __reorder_size = __calc_reorder_slm_size();
        constexpr std::uint32_t __slm_size =
            std::max(__work_item_all_hists_size, __reorder_size) + __group_hist_size + __global_hist_size;

        return __slm_size;
    }

    const _GlobOffsetT __n;
    const std::uint32_t __stage;
    _GlobOffsetT* __p_global_hist;
    _GlobOffsetT* __p_group_hists;
    _InRngPack __in_pack;
    _OutRngPack __out_pack;
    sycl::local_accessor<unsigned char, 1> __slm_accessor;
    std::uint32_t __num_tiles;

    __radix_sort_onesweep_kernel(_GlobOffsetT __n, std::uint32_t __stage, _GlobOffsetT* __p_global_hist,
                                 _GlobOffsetT* __p_group_hists, const _InRngPack& __in_pack,
                                 const _OutRngPack& __out_pack, sycl::local_accessor<unsigned char, 1> __slm_acc,
                                 std::uint32_t __num_tiles)
        : __n(__n), __stage(__stage), __p_global_hist(__p_global_hist), __p_group_hists(__p_group_hists),
          __in_pack(__in_pack), __out_pack(__out_pack), __slm_accessor(__slm_acc), __num_tiles(__num_tiles)
    {
    }

    template <typename _KVPack>
    inline auto
    __load_pack(_KVPack& __pack, std::uint32_t __tile_id, std::uint32_t __sg_id, std::uint32_t __sg_local_id) const
    {
        const _GlobOffsetT __offset = __data_per_sub_group * (__tile_id * __num_sub_groups_per_work_group + __sg_id);
        auto __keys_seq = __rng_data(__in_pack.__keys_rng());
        __load</*__sort_identity_residual=*/true>(__pack.__keys, __keys_seq, __offset, __sg_local_id);
        if constexpr (__has_values)
        {
            __load</*__sort_identity_residual=*/false>(__pack.__vals, __rng_data(__in_pack.__vals_rng()), __offset,
                                                       __sg_local_id);
        }
    }

    template <bool __sort_identity_residual, typename _T, typename _InSeq>
    inline void
    __load(_T __elements[__data_per_work_item], const _InSeq& __in_seq, _GlobOffsetT __glob_offset,
           std::uint32_t __local_offset) const
    {
        bool __is_full_block = (__glob_offset + __data_per_sub_group) <= __n;
        _GlobOffsetT __offset = __glob_offset + __local_offset;
        if (__is_full_block)
        {
            _ONEDPL_PRAGMA_UNROLL
            for (std::uint32_t __i = 0; __i < __data_per_work_item; ++__i)
            {
                __elements[__i] = __in_seq[__offset + __i * __sub_group_size];
            }
        }
        else
        {
            _ONEDPL_PRAGMA_UNROLL
            for (std::uint32_t __i = 0; __i < __data_per_work_item; ++__i)
            {
                _GlobOffsetT __idx = __offset + __i * __sub_group_size;
                if constexpr (__sort_identity_residual)
                {
                    __elements[__i] = (__idx < __n) ? __in_seq[__idx] : __sort_identity<_T, __is_ascending>();
                }
                else
                {
                    if (__idx < __n)
                    {
                        __elements[__i] = __in_seq[__idx];
                    }
                }
            }
        }
    }

    static inline std::uint32_t
    __match_bins(sycl::sub_group __sub_group, std::uint32_t __bin)
    {
        // start with all bits 1
        sycl::ext::oneapi::sub_group_mask __matched_bins = sycl::ext::oneapi::group_ballot(__sub_group);
        _ONEDPL_PRAGMA_UNROLL
        for (int __i = 0; __i < __radix_bits; __i++)
        {
            bool __bit = static_cast<bool>((__bin >> __i) & 1);
            sycl::ext::oneapi::sub_group_mask __sg_vote = sycl::ext::oneapi::group_ballot(__sub_group, __bit);
            // If we vote yes, then we want to set all bits that also voted yes. If no, then we want to
            // zero out the bits that said yes as they don't match and preserve others as we have no info on these.
            __matched_bins &= __bit ? __sg_vote : ~__sg_vote;
        }
        std::uint32_t __result = 0;
        __matched_bins.extract_bits(__result);
        return __result;
    }

    inline auto
    __rank_local(const sycl::nd_item<1>& __idx, sycl::sub_group __sub_group, _LocOffsetT __ranks[__data_per_work_item],
                 _LocOffsetT __bins[__data_per_work_item], _LocOffsetT* __slm_subgroup_hists,
                 std::uint32_t __sub_group_slm_offset, std::uint32_t __sub_group_local_id) const
    {
        _LocOffsetT* __slm_offset = __slm_subgroup_hists + __sub_group_slm_offset;

        for (_LocIdxT __i = __sub_group_local_id; __i < __bin_count; __i += __sub_group_size)
        {
            __slm_offset[__i] = 0;
        }

        constexpr _SubGroupBitmaskT __sub_group_full_bitmask = 0x7fffffff;
        static_assert(__sub_group_size == 32);
        // lower bits than my current will be set meaning we only preserve left lanes
        _SubGroupBitmaskT __remove_right_lanes =
            __sub_group_full_bitmask >> (__sub_group_size - 1 - __sub_group_local_id);

        _ONEDPL_PRAGMA_UNROLL
        for (std::uint32_t __i = 0; __i < __data_per_work_item; ++__i)
        {
            _LocOffsetT __bin = __bins[__i];
            _SubGroupBitmaskT __matched_bins = __match_bins(__sub_group, __bin);
            sycl::group_barrier(__sub_group);
            _LocOffsetT __pre_rank = __slm_offset[__bin];
            _SubGroupBitmaskT __matched_left_lanes = __matched_bins & __remove_right_lanes;
            _LocOffsetT __this_round_rank = sycl::popcount(__matched_left_lanes);
            _LocOffsetT __this_round_count = sycl::popcount(__matched_bins);
            _LocOffsetT __rank_after = __pre_rank + __this_round_rank;
            bool __is_leader = __this_round_rank == __this_round_count - 1;
            sycl::group_barrier(__sub_group);
            if (__is_leader)
            {
                __slm_offset[__bin] = __rank_after + 1;
            }
            __ranks[__i] = __rank_after;
        }
        __dpl_sycl::__group_barrier(__idx);
    }

    inline void
    __rank_global(const sycl::nd_item<1>& __idx, sycl::sub_group __sub_group, std::uint32_t __tile_id,
                  std::uint32_t __sub_group_id, std::uint32_t __sub_group_local_id, _LocOffsetT* __slm_subgroup_hists,
                  _LocOffsetT* __slm_group_hist, _GlobOffsetT* __slm_global_incoming) const
    {
        // TODO: This exists in the ESIMD KT and was ported but are we not limiting max input size to
        // 2^30 ~ 1 billion elements? We use 32-bit indexing / histogram which may already be too small
        // but are then reserving the two upper bits for lookback flags.
        constexpr std::uint32_t __global_accumulated = 0x40000000;
        constexpr std::uint32_t __hist_updated = 0x80000000;
        constexpr std::uint32_t __global_offset_mask = 0x3fffffff;

        _GlobOffsetT* __p_this_group_hist = __p_group_hists + __bin_count * __tile_id;
        _GlobOffsetT* __p_prev_group_hist = __p_this_group_hist - __bin_count;

        // This is important so that we can evenly partition the radix bits across a number of sub-groups
        // without masking lanes. Radix bits is always a power of two, so this requirement essentially just
        // requires radix_bits >= 5 for sub-group size of 32.
        static_assert(__bin_count % __sub_group_size == 0);

        constexpr std::uint32_t __bin_summary_sub_group_size = __bin_count / __sub_group_size;
        constexpr std::uint32_t __bin_process_width = __sub_group_size;

        // 1. Vector scan of histograms previously accumulated by each work-item
        // update slm instead of grf summary due to perf issues with grf histogram

        // TODO: this single element array is a temporary workaround for sub group scan requiring an array
        _LocOffsetT __item_grf_hist_summary_arr[1] = {0};
        _LocOffsetT& __item_grf_hist_summary = __item_grf_hist_summary_arr[0];
        _LocOffsetT __item_bin_count = 0;
        if (__sub_group_id < __bin_summary_sub_group_size)
        {
            // 1.1. Vector scan of the same bins across different histograms. Each lane is assigned its own bin and
            // scans across all sub-group histograms.
            __item_bin_count = __intra_bin_scan_across_sub_groups<__bin_process_width>(
                __sub_group_id, __sub_group_local_id, __item_grf_hist_summary, __slm_subgroup_hists);

            // 1.2. Vector scan of different bins inside one histogram: ONLY the final one per summary sub-group
            __inter_bin_scan_work_group_totals<__bin_process_width>(__sub_group, __sub_group_id, __sub_group_local_id,
                                                                    __item_grf_hist_summary_arr, __slm_group_hist);

            // 1.3. Copy the histogram at the region designated for synchronization between work-groups and set work-group
            // zeros incoming values from the global histogram kernel.
            __output_work_group_chained_scan_partials<__bin_process_width, __global_accumulated, __hist_updated,
                                                      __global_offset_mask>(__tile_id, __sub_group_id,
                                                                            __sub_group_local_id, __item_bin_count,
                                                                            __p_this_group_hist, __slm_global_incoming);
        }
        __dpl_sycl::__group_barrier(__idx);

        // 1.4 One work-item finalizes scan performed at stage 1.2
        // by propagating prefixes accumulated after scanning individual "__bin_process_width" pieces and converting
        // them scan from being inclusive to exclusive.
        if (__sub_group_id == 0)
        {
            __sub_group_cross_segment_exclusive_scan<__bin_process_width, __bin_summary_sub_group_size>(
                __sub_group, __sub_group_local_id, __slm_group_hist);
        }

        __dpl_sycl::__group_barrier(__idx);

        // 2. Chained scan. Synchronization between work-groups.
        if (__sub_group_id < __bin_summary_sub_group_size && __tile_id != 0)
        {
            __work_group_chained_scan<__bin_process_width, __bin_summary_sub_group_size, __global_accumulated,
                                      __hist_updated, __global_offset_mask>(__idx, __sub_group, __sub_group_local_id,
                                                                            __item_bin_count, __p_this_group_hist,
                                                                            __p_prev_group_hist, __slm_global_incoming);
        }

        __dpl_sycl::__group_barrier(__idx);
    }

    template <std::uint32_t __bin_process_width>
    inline _LocOffsetT
    __intra_bin_scan_across_sub_groups(std::uint32_t __sub_group_id, std::uint32_t __sub_group_local_id,
                                       _LocOffsetT& __item_grf_hist_summary, _LocOffsetT* __slm_subgroup_hists) const
    {
        _LocIdxT __slm_bin_hist_summary_offset = __sub_group_id * __bin_process_width;

        for (std::uint32_t __s = 0; __s < __num_sub_groups_per_work_group;
             __s++, __slm_bin_hist_summary_offset += __bin_count)
        {
            _LocIdxT __slm_idx = __slm_bin_hist_summary_offset + __sub_group_local_id;
            __item_grf_hist_summary += __slm_subgroup_hists[__slm_idx];
            __slm_subgroup_hists[__slm_idx] = __item_grf_hist_summary;
        }
        return __item_grf_hist_summary;
    }

    template <std::uint32_t __bin_process_width>
    inline void
    __inter_bin_scan_work_group_totals(sycl::sub_group __sub_group, std::uint32_t __sub_group_id,
                                       std::uint32_t __sub_group_local_id,
                                       _LocOffsetT (&__item_grf_hist_summary_arr)[1],
                                       _LocOffsetT* __slm_group_hist) const
    {
        __sub_group_scan<__sub_group_size, 1>(__sub_group, __item_grf_hist_summary_arr, std::plus<>{},
                                              __bin_process_width);

        _LocIdxT __write_idx = __sub_group_id * __bin_process_width + __sub_group_local_id;
        __slm_group_hist[__write_idx] = __item_grf_hist_summary_arr[0];
    }

    template <std::uint32_t __segment_width, std::uint32_t __num_segments, typename _ScanBuffer>
    void
    __sub_group_cross_segment_exclusive_scan(sycl::sub_group& __sub_group, std::uint32_t __sub_group_local_id,
                                             _ScanBuffer* __scan_elements) const
    {
        // __segment_width is required to match __sub_group_size for performance: each lane processes
        // one element and no masking is required. However to support radix bits < log2(sub_group_size)
        // we would need to relax this requirement and add masking with a new, higher overhead path
        static_assert(__segment_width == __sub_group_size);
        using _ElemT = std::remove_reference_t<decltype(__scan_elements[0])>;
        _ElemT __carry = 0;

        _ONEDPL_PRAGMA_UNROLL
        for (std::uint32_t __i = 0; __i < __num_segments; ++__i)
        {
            _ElemT __element = __scan_elements[__i * __segment_width + __sub_group_local_id];
            _ElemT __element_right_shift = sycl::shift_group_right(__sub_group, __element, 1);
            if (__sub_group_local_id == 0)
                __element_right_shift = 0;
            __scan_elements[__i * __segment_width + __sub_group_local_id] = __element_right_shift + __carry;

            __carry += sycl::group_broadcast(__sub_group, __element, __sub_group_size - 1);
        }
    }

    template <std::uint32_t __bin_process_width, std::uint32_t __global_accumulated, std::uint32_t __hist_updated,
              std::uint32_t __global_offset_mask>
    inline void
    __output_work_group_chained_scan_partials(std::uint32_t __tile_id, std::uint32_t __sub_group_id,
                                              std::uint32_t __sub_group_local_id, _LocOffsetT __item_bin_count,
                                              _GlobOffsetT* __p_this_group_hist,
                                              _GlobOffsetT* __slm_global_incoming) const
    {
        using _GlobalAtomicT = sycl::atomic_ref<_GlobOffsetT, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                                sycl::access::address_space::global_space>;
        _LocIdxT __hist_idx = __sub_group_id * __bin_process_width + __sub_group_local_id;

        if (__tile_id != 0)
        {
            // Copy the histogram, local to this WG
            _GlobalAtomicT __ref(__p_this_group_hist[__hist_idx]);
            __ref.store(__item_bin_count | __hist_updated);
        }
        else
        {
            // WG0 is a special case: it also retrieves the total global histogram and adds it to its local histogram
            // This global histogram will be propagated to other work-groups through a chained scan at stage 2
            _GlobOffsetT __global_hist = __p_global_hist[__hist_idx] & __global_offset_mask;
            _GlobOffsetT __after_group_hist_sum = __global_hist + __item_bin_count;
            _GlobalAtomicT __ref(__p_this_group_hist[__hist_idx]);
            __ref.store(__after_group_hist_sum | __hist_updated | __global_accumulated);
            // Copy the global histogram to local memory to share with other work-items
            __slm_global_incoming[__hist_idx] = __global_hist;
        }
    }

    template <std::uint32_t __bin_process_width, std::uint32_t __bin_summary_sub_group_size,
              std::uint32_t __global_accumulated, std::uint32_t __hist_updated, std::uint32_t __global_offset_mask>
    inline void
    __work_group_chained_scan(const sycl::nd_item<1>& __idx, sycl::sub_group __sub_group,
                              std::uint32_t __sub_group_local_id, _LocOffsetT __item_bin_count,
                              _GlobOffsetT* __p_this_group_hist, _GlobOffsetT* __p_prev_group_hist,
                              _GlobOffsetT* __slm_global_incoming) const
    {
        using _GlobalAtomicT = sycl::atomic_ref<_GlobOffsetT, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                                sycl::access::address_space::global_space>;
        std::uint32_t __sub_group_group_id = __sub_group.get_group_linear_id();

        // 2.1. Read the histograms scanned across work-groups
        _GlobOffsetT __prev_group_hist_sum = 0;
        _GlobOffsetT __prev_group_hist;
        bool __is_not_accumulated = true;
        _GlobOffsetT* __p_lookback_hist = __p_prev_group_hist;
        do
        {
            // On Xe2, we have seen some low probability instances where the lookback gets stuck when using relaxed atomic loads even though
            // lower work-groups have written. Using a higher memory order for the atomic has a very high performance cost. To mitigate
            // this, we execute an acquire atomic fence after __atomic_fence_iter iterations to unblock any stalled items and
            // between all tile iterations we execute a release atomic fence. As this stalling issue seldom occurs, the performance impact from
            // this memory order is small, and we maintain safety.
            constexpr std::uint32_t __atomic_fence_iter = 256;
            std::uint32_t __lookback_counter = 0;
            _LocIdxT __bin_idx = __sub_group_group_id * __bin_process_width + __sub_group_local_id;
            _GlobalAtomicT __ref(__p_lookback_hist[__bin_idx]);
            do
            {
                __prev_group_hist =
                    (__lookback_counter < __atomic_fence_iter) ? __ref.load() : __ref.load(sycl::memory_order::acquire);
                ++__lookback_counter;
            } while ((__prev_group_hist & __hist_updated) == 0);
            __prev_group_hist_sum += __is_not_accumulated ? __prev_group_hist : 0;
            __is_not_accumulated = (__prev_group_hist_sum & __global_accumulated) == 0;
            __p_lookback_hist -= __bin_count;
        } while (sycl::any_of_group(__sub_group, __is_not_accumulated));

        __prev_group_hist_sum &= __global_offset_mask;
        _GlobOffsetT __after_group_hist_sum = __prev_group_hist_sum + __item_bin_count;
        _LocIdxT __bin_idx = __sub_group_group_id * __bin_process_width + __sub_group_local_id;

        // 2.2. Write the histogram scanned across work-group, updated with the current work-group data
        _GlobalAtomicT __ref(__p_this_group_hist[__bin_idx]);
        __ref.store(__after_group_hist_sum | __hist_updated | __global_accumulated);

        // 2.3. Save the scanned histogram from previous work-groups locally
        __slm_global_incoming[__bin_idx] = __prev_group_hist_sum;
    }

    void inline __propagate_ranks_across_sub_groups(_LocOffsetT (&__ranks)[__data_per_work_item],
                                                    const _LocOffsetT (&__bins)[__data_per_work_item],
                                                    _LocOffsetT* __slm_subgroup_hists, _LocOffsetT* __slm_group_hist,
                                                    std::uint32_t __sub_group_id) const
    {
        // update ranks to reflect sub-group offsets in and across bins
        _ONEDPL_PRAGMA_UNROLL
        for (std::uint32_t __i = 0; __i < __data_per_work_item; ++__i)
        {
            _LocOffsetT __bin = __bins[__i];
            _LocOffsetT __group_incoming = __slm_group_hist[__bin];
            _LocOffsetT __offset_in_bin =
                (__sub_group_id == 0) ? 0 : __slm_subgroup_hists[(__sub_group_id - 1) * __bin_count + __bin];
            _LocOffsetT __offset_across_bins = __group_incoming;
            __ranks[__i] += __offset_in_bin + __offset_across_bins;
        }
    }

    void inline __global_fix_to_slm(const sycl::nd_item<1>& __idx, _GlobOffsetT* __slm_global_incoming,
                                    _LocOffsetT* __slm_group_hist) const
    {
        // When we reorder into SLM there are indexing offsets between bins due to contiguous storage that should not be reflected in global output as any given bin's
        // total global offset is defined in __slm_global_incoming. We account for this by subtracting each bin's incoming slm index offset
        // from __slm_global_incoming so that later adding the reorderered key's slm index to the fixed global offset yields the correct output index in the final stage.
        //
        //
        // The sequence of computations for the fixed global offset is shown below, showing how we yield a valid output index in __reorder_slm_to_glob.
        // For demonstration, slm_global_fix is separated from slm_global_incoming which can actually be modified in-place.
        // slm_global_fix[bin] = slm_global_incoming[bin] - slm_group_hist[bin]
        // slm_idx[key]        = slm_group_hist[bin] + key offset within bin
        // out_idx[key]        = slm_global_fix[bin] + slm_idx[key]
        //                     = slm_global_incoming[bin] - slm_group_hist[bin] + slm_group_hist[bin] + key offset within bin
        //                     = slm_global_incoming[bin] + key offset within bin
        //
        // The case where __slm_group_hist[_i] > __slm_global_incoming[__i] is valid resulting in
        // the difference yielding a large number due to guaranteed wrap around behavior with unsigned integers in the C++ spec.
        // When this global fix is added to the reordered offset index the wraparound is undone, yielding the valid output index shown above.
        for (_LocIdxT __i = __idx.get_local_id(); __i < __bin_count; __i += __work_group_size)
        {
            __slm_global_incoming[__i] -= __slm_group_hist[__i];
        }
        __dpl_sycl::__group_barrier(__idx);
    }

    template <typename _KVPack>
    void inline __reorder_reg_to_slm(const sycl::nd_item<1>& __idx, const _KVPack& __pack,
                                     _LocOffsetT (&__ranks)[__data_per_work_item],
                                     const _LocOffsetT (&__bins)[__data_per_work_item], std::uint32_t __sub_group_id,
                                     _LocOffsetT* __slm_subgroup_hists, _LocOffsetT* __slm_group_hist,
                                     _GlobOffsetT* __slm_global_incoming, _KeyT* __slm_keys, _ValT* __slm_vals) const
    {
        // 1. update ranks to reflect sub-group offsets in and across bins
        __propagate_ranks_across_sub_groups(__ranks, __bins, __slm_subgroup_hists, __slm_group_hist, __sub_group_id);

        // 2. Apply fix to __slm_global_incoming
        __global_fix_to_slm(__idx, __slm_global_incoming, __slm_group_hist);

        // 3. Write keys (and values) to SLM at computed ranks
        _ONEDPL_PRAGMA_UNROLL
        for (std::uint32_t __i = 0; __i < __data_per_work_item; ++__i)
        {
            __slm_keys[__ranks[__i]] = __pack.__keys[__i];
            if constexpr (__has_values)
            {
                __slm_vals[__ranks[__i]] = __pack.__vals[__i];
            }
        }
        __dpl_sycl::__group_barrier(__idx);
    }

    template <typename _KVPack>
    void inline __reorder_slm_to_glob(const sycl::nd_item<1>& __idx, _KVPack& __pack, std::uint32_t __sub_group_id,
                                      std::uint32_t __sub_group_local_id, _GlobOffsetT* __slm_global_fix,
                                      _KeyT* __slm_keys, _ValT* __slm_vals) const
    {

        const _GlobOffsetT __keys_slm_offset = __data_per_sub_group * __sub_group_id;

        _ONEDPL_PRAGMA_UNROLL
        for (std::uint32_t __i = 0; __i < __data_per_work_item; ++__i)
        {
            _LocIdxT __slm_idx = __keys_slm_offset + __i * __sub_group_size + __sub_group_local_id;
            _KeyT __key = __slm_keys[__slm_idx];
            _LocIdxT __bin = __get_bucket_scalar<__mask>(__order_preserving_cast_scalar<__is_ascending>(__key),
                                                         __stage * __radix_bits);
            _GlobOffsetT __global_fix = __slm_global_fix[__bin];
            _GlobOffsetT __out_idx = __global_fix + __slm_idx;

            // TODO: we need to figure out how to relax this bounds checking for full unrolling
            bool __output_mask = __out_idx < __n;
            if (__output_mask)
                __out_pack.__keys_rng()[__out_idx] = __key;
            if constexpr (__has_values)
            {
                _ValT __val = __slm_vals[__slm_idx];
                if (__output_mask)
                    __out_pack.__vals_rng()[__out_idx] = __val;
            }
        }
    }

    auto
    get(syclex::properties_tag) const
    {
        return syclex::properties{syclex::work_group_progress<syclex::forward_progress_guarantee::concurrent,
                                                              syclex::execution_scope::root_group>,
                                  syclex::sub_group_size<32>};
    }

    void
    operator()(sycl::nd_item<1> __idx) const
    {
        sycl::sub_group __sub_group = __idx.get_sub_group();
        const std::uint32_t __sg_id = __sub_group.get_group_linear_id();
        const std::uint32_t __sg_local_id = __sub_group.get_local_id();

        const std::uint32_t __sub_group_slm_offset = __sg_id * __bin_count;
        std::uint32_t __tile_id = __idx.get_group().get_group_linear_id();
        std::uint32_t __num_wgs = __idx.get_group_range(0);
        for (; __tile_id < __num_tiles; __tile_id += __num_wgs)
        {
            auto __values_pack = __make_key_value_pack<__data_per_work_item, _KeyT, _ValT>();
            _LocOffsetT __bins[__data_per_work_item];
            _LocOffsetT __ranks[__data_per_work_item];

            __load_pack(__values_pack, __tile_id, __sg_id, __sg_local_id);

            _ONEDPL_PRAGMA_UNROLL
            for (std::uint32_t __i = 0; __i < __data_per_work_item; ++__i)
            {
                const auto __ordered = __order_preserving_cast_scalar<__is_ascending>(__values_pack.__keys[__i]);
                __bins[__i] = __get_bucket_scalar<__mask>(__ordered, __stage * __radix_bits);
            }

            // Get raw SLM pointer and create typed pointers for different regions using helper functions
            unsigned char* __slm_raw = __slm_accessor.get_multi_ptr<sycl::access::decorated::no>().get();
            _LocOffsetT* __slm_subgroup_hists = reinterpret_cast<_LocOffsetT*>(__slm_raw);
            _LocOffsetT* __slm_group_hist = reinterpret_cast<_LocOffsetT*>(__slm_raw + __get_slm_group_hist_offset());
            _GlobOffsetT* __slm_global_incoming =
                reinterpret_cast<_GlobOffsetT*>(__slm_raw + __get_slm_global_incoming_offset());

            __rank_local(__idx, __sub_group, __ranks, __bins, __slm_subgroup_hists, __sub_group_slm_offset,
                         __sg_local_id);
            __rank_global(__idx, __sub_group, __tile_id, __sg_id, __sg_local_id, __slm_subgroup_hists, __slm_group_hist,
                          __slm_global_incoming);

            // For reorder phase, reinterpret the sub-group histogram space as key/value storage
            // The reorder space overlaps with the sub-group histogram region (reinterpret_cast)
            _KeyT* __slm_keys = reinterpret_cast<_KeyT*>(__slm_raw);
            _ValT* __slm_vals = nullptr;
            if constexpr (__has_values)
            {
                __slm_vals =
                    reinterpret_cast<_ValT*>(__slm_raw + __work_group_size * __data_per_work_item * sizeof(_KeyT));
            }

            __reorder_reg_to_slm(__idx, __values_pack, __ranks, __bins, __sg_id, __slm_subgroup_hists, __slm_group_hist,
                                 __slm_global_incoming, __slm_keys, __slm_vals);

            __reorder_slm_to_glob(__idx, __values_pack, __sg_id, __sg_local_id, __slm_global_incoming, __slm_keys,
                                  __slm_vals);

            sycl::group_barrier(__idx.get_group());
            // Make sure our atomic updates are pushed to other groups
            sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::device);
        }
    }
};

} // namespace oneapi::dpl::experimental::kt::gpu::__impl

#endif // _ONEDPL_KT_SYCL_RADIX_SORT_KERNELS_H
