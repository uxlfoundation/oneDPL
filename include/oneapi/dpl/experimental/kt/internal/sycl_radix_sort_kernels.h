// -*- C++ -*-
//===-- sycl_radix_sort_kernels.h ----------------------------------------===//
//
// Copyright (C) UXL Foundation Contributors
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
          std::uint16_t __hist_work_group_size, typename _KeysRng, typename _ClearPack1, typename _ClearPack2>
struct __global_histogram_sycl
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
    // Scatter destinations zeroed while reading the keys, see __radix_sort_onesweep_kernel::__clear_pack
    _ClearPack1 __clear_pack1;
    _ClearPack2 __clear_pack2;
    bool __clear1;
    bool __clear2;

    __global_histogram_sycl(std::size_t __n, const _KeysRng& __keys_rng,
                            sycl::local_accessor<std::uint32_t, 1> __slm_acc, std::uint32_t* __p_global_offset,
                            std::uint32_t __num_histograms, const _ClearPack1& __clear_pack1,
                            const _ClearPack2& __clear_pack2, bool __clear1, bool __clear2)
        : __n(__n), __keys_rng(__keys_rng), __slm_acc(__slm_acc), __p_global_offset(__p_global_offset),
          __num_histograms(__num_histograms), __clear_pack1(__clear_pack1), __clear_pack2(__clear_pack2),
          __clear1(__clear1), __clear2(__clear2)
    {
    }

    template <typename _Pack>
    static void
    __clear_range(const _Pack& __pack, std::size_t __start, std::size_t __count, std::uint32_t __lane)
    {
        __zero_range(&__rng_data(__pack.__keys_rng())[__start], __count, __lane, __sub_group_size);
        if constexpr (std::decay_t<_Pack>::__has_values)
        {
            __zero_range(&__rng_data(__pack.__vals_rng())[__start], __count, __lane, __sub_group_size);
        }
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

        sycl::group_barrier(__idx.get_group());

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
            if (__clear1 || __clear2)
            {
                const std::size_t __chunk_start = __wi_offset - __sub_group_local_id;
                const std::size_t __chunk_count = std::min<std::size_t>(__n - __chunk_start, __hist_data_per_sub_group);
                if (__clear1)
                    __clear_range(__clear_pack1, __chunk_start, __chunk_count, __sub_group_local_id);
                if (__clear2)
                    __clear_range(__clear_pack2, __chunk_start, __chunk_count, __sub_group_local_id);
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
                        oneapi::dpl::__internal::__order_preserving_cast<__is_ascending>(__keys[__i]),
                        __stage * __radix_bits);
                    _GlobOffsetT __bin = __stage * __bin_count + __bucket;
                    using _SLMAtomicRef =
                        sycl::atomic_ref<_GlobOffsetT, sycl::memory_order::relaxed, sycl::memory_scope::work_group,
                                         sycl::access::address_space::local_space>;
                    _SLMAtomicRef __slm_ref(__slm[__bin * __num_histograms + __slm_hist_lane_offset]);
                    __slm_ref.fetch_add(1);
                }
            }
        }

        sycl::group_barrier(__idx.get_group());

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

    // TODO: The same limitation exists in the ESIMD KT but these masks limit max input size to
    // 2^30 ~ 1 billion elements.
    static constexpr _GlobOffsetT __global_accumulated_mask = 0x40000000;
    static constexpr _GlobOffsetT __hist_updated_mask = 0x80000000;
    static constexpr _GlobOffsetT __global_offset_mask = 0x3fffffff;

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
        // Followed by the segment totals of the cross-bin scan (__bin_count / __sub_group_size values, 32 bytes
        // reserved), and the last 4 bytes hold the id of the tile the work-group processes next
        constexpr std::uint32_t __slm_size = std::max(__work_item_all_hists_size, __reorder_size) +
                                             __group_hist_size + __global_hist_size + 32 + sizeof(std::uint32_t);

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
    // Zero the input after loading it, see __clear_pack
    bool __clear_input;

    __radix_sort_onesweep_kernel(_GlobOffsetT __n, std::uint32_t __stage, _GlobOffsetT* __p_global_hist,
                                 _GlobOffsetT* __p_group_hists, const _InRngPack& __in_pack,
                                 const _OutRngPack& __out_pack, sycl::local_accessor<unsigned char, 1> __slm_acc,
                                 std::uint32_t __num_tiles, bool __clear_input)
        : __n(__n), __stage(__stage), __p_global_hist(__p_global_hist), __p_group_hists(__p_group_hists),
          __in_pack(__in_pack), __out_pack(__out_pack), __slm_accessor(__slm_acc), __num_tiles(__num_tiles),
          __clear_input(__clear_input)
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

    // The input of a stage is the output of the stage after next, so it is zeroed after loading to prepare the
    // later scatter.
    inline void
    __clear_pack(std::uint32_t __tile_id, std::uint32_t __sg_id, std::uint32_t __sg_local_id) const
    {
        const _GlobOffsetT __offset = __data_per_sub_group * (__tile_id * __num_sub_groups_per_work_group + __sg_id);
        __clear(__rng_data(__in_pack.__keys_rng()), __offset, __sg_local_id);
        if constexpr (__has_values)
        {
            __clear(__rng_data(__in_pack.__vals_rng()), __offset, __sg_local_id);
        }
    }

    template <typename _OutSeq>
    inline void
    __clear(_OutSeq&& __out_seq, _GlobOffsetT __glob_offset, std::uint32_t __local_offset) const
    {
        if (__glob_offset < __n)
            __zero_range(&__out_seq[__glob_offset], std::min<std::size_t>(__n - __glob_offset, __data_per_sub_group),
                         __local_offset, __sub_group_size);
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
            // With one pointer per work-item the element offsets are constants: the compiler does not compute a 64-bit
            // address from a 32-bit index for each element
            const _T* __p = &__in_seq[__offset];
            _ONEDPL_PRAGMA_UNROLL
            for (std::uint32_t __i = 0; __i < __data_per_work_item; ++__i)
            {
                __elements[__i] = __p[__i * __sub_group_size];
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

    // The bin of a key in this pass. It is recomputed where it is needed instead of being kept in an array: with large
    // tiles, the bins and ranks of all keys kept alive across the tile's scans do not fit in the registers.
    inline std::uint32_t
    __key_bin(_KeyT __key) const
    {
        return __get_bucket_scalar<__mask>(oneapi::dpl::__internal::__order_preserving_cast<__is_ascending>(__key),
                                           __stage * __radix_bits);
    }

    // The same value as __key_bin, written differently on purpose: otherwise the compiler reuses the bins computed for
    // the ranking and keeps them alive across the tile's scans, which causes register spills with large tiles
    inline std::uint32_t
    __key_bin_after_scans(_KeyT __key) const
    {
        const auto __ordered = oneapi::dpl::__internal::__order_preserving_cast<__is_ascending>(__key);
        constexpr std::uint32_t __ordered_bits = sizeof(__ordered) * 8;
        return static_cast<std::uint32_t>((__ordered << (__ordered_bits - __radix_bits - __stage * __radix_bits)) >>
                                          (__ordered_bits - __radix_bits));
    }

    static inline std::uint32_t
    __match_bins(sycl::sub_group __sub_group, std::uint32_t __bin)
    {
        // For each bit, keep the lanes that voted like this lane: bit ? vote : ~vote, computed as vote ^ (bit - 1).
        // Working on 32-bit values avoids the wider sub_group_mask arithmetic.
        std::uint32_t __matched_bins = ~std::uint32_t{0};
        _ONEDPL_PRAGMA_UNROLL
        for (int __i = 0; __i < __radix_bits; __i++)
        {
            const std::uint32_t __bit = (__bin >> __i) & 1;
            std::uint32_t __vote = 0;
            sycl::ext::oneapi::group_ballot(__sub_group, __bit != 0).extract_bits(__vote);
            __matched_bins &= __vote ^ (__bit - 1);
        }
        return __matched_bins;
    }

    inline auto
    __rank_local(const sycl::nd_item<1>& __idx, sycl::sub_group __sub_group, _LocOffsetT __ranks[__data_per_work_item],
                 const _KeyT (&__keys)[__data_per_work_item], _LocOffsetT* __slm_subgroup_hists,
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
            _LocOffsetT __bin = __key_bin(__keys[__i]);
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
        sycl::group_barrier(__idx.get_group());
    }

    // Scans the tile's histograms and publishes its aggregate. Returns the tile's count of this lane's bin
    // (lanes of the first __bin_count / __sub_group_size sub-groups). The lookback is done by __rank_global_lookback.
    inline _LocOffsetT
    __rank_global(const sycl::nd_item<1>& __idx, sycl::sub_group __sub_group, std::uint32_t __tile_id,
                  std::uint32_t __sub_group_id, std::uint32_t __sub_group_local_id, _LocOffsetT* __slm_subgroup_hists,
                  _LocOffsetT* __slm_group_hist, _GlobOffsetT* __slm_global_incoming) const
    {
        auto __group = __idx.get_group();
        // Small SLM area after the global offsets, see __calc_slm_alloc
        _LocOffsetT* __slm_segment_totals = reinterpret_cast<_LocOffsetT*>(__slm_global_incoming + __bin_count);
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

            // 1.2. Copy the histogram at the region designated for synchronization between work-groups and set work-group
            // incoming values from the global histogram kernel.
            __output_work_group_chained_scan_partials<__bin_process_width>(__tile_id, __sub_group_id,
                                                                           __sub_group_local_id, __item_bin_count,
                                                                           __p_this_group_hist, __slm_global_incoming);

            // 1.3. Partial scan across bins: Each participating sub-group independently scans its own
            // segment of __bin_process_width bins. These partial results are finalized in step 1.4.
            __sub_group_scan<__sub_group_size, 1>(__idx, __item_grf_hist_summary_arr, std::plus<>{},
                                                  __bin_process_width);
            // The last lane holds the total of this segment of __bin_process_width bins
            if (__sub_group_local_id == __sub_group_size - 1)
                __slm_segment_totals[__sub_group_id] = __item_grf_hist_summary_arr[0];
        }
        sycl::group_barrier(__group);

        // 1.4. Exclusive scan across bins: each sub-group adds the totals of the earlier segments to its own
        // inclusive segment scan (kept in registers) and converts it to exclusive.
        if (__sub_group_id < __bin_summary_sub_group_size)
        {
            _LocOffsetT __carry = 0;
            _ONEDPL_PRAGMA_UNROLL
            for (std::uint32_t __k = 0; __k < __bin_summary_sub_group_size; ++__k)
            {
                if (__k < __sub_group_id)
                    __carry += __slm_segment_totals[__k];
            }
            __slm_group_hist[__sub_group_id * __bin_process_width + __sub_group_local_id] =
                __item_grf_hist_summary_arr[0] - __item_bin_count + __carry;
        }

        sycl::group_barrier(__group);
        return __item_bin_count;
    }

    // 2. Chained scan: synchronization between work-groups. Stores, per bin, the global offset of the bin's first key
    // minus the bin's offset in the reordered tile, so the scatter computes the output index as this value plus the
    // key's SLM index (see __reorder_slm_to_glob).
    inline void
    __rank_global_lookback(const sycl::nd_item<1>& __idx, sycl::sub_group __sub_group, std::uint32_t __tile_id,
                           std::uint32_t __sub_group_id, std::uint32_t __sub_group_local_id,
                           _LocOffsetT __item_bin_count, _LocOffsetT* __slm_group_hist,
                           _GlobOffsetT* __slm_global_incoming) const
    {
        constexpr std::uint32_t __bin_summary_sub_group_size = __bin_count / __sub_group_size;
        constexpr std::uint32_t __bin_process_width = __sub_group_size;
        if (__sub_group_id < __bin_summary_sub_group_size)
        {
            if (__tile_id != 0)
            {
                _GlobOffsetT* __p_this_group_hist = __p_group_hists + __bin_count * __tile_id;
                __work_group_chained_scan<__bin_process_width, __bin_summary_sub_group_size>(
                    __idx, __sub_group, __sub_group_local_id, __item_bin_count, __p_this_group_hist,
                    __p_this_group_hist - __bin_count, __slm_global_incoming);
            }
            // The same lane wrote this bin's global offset (lookback, or the global histogram for tile 0)
            const _LocIdxT __bin = __sub_group_id * __bin_process_width + __sub_group_local_id;
            __slm_global_incoming[__bin] -= __slm_group_hist[__bin];
        }
    }

    template <std::uint32_t __bin_process_width>
    inline _LocOffsetT
    __intra_bin_scan_across_sub_groups(std::uint32_t __sub_group_id, std::uint32_t __sub_group_local_id,
                                       _LocOffsetT& __item_grf_hist_summary, _LocOffsetT* __slm_subgroup_hists) const
    {
        const _LocIdxT __slm_idx = __sub_group_id * __bin_process_width + __sub_group_local_id;
        // Independent loads first, then the running sum in registers, so the SLM accesses are not serialized
        _LocOffsetT __counts[__num_sub_groups_per_work_group];
        _ONEDPL_PRAGMA_UNROLL
        for (std::uint32_t __s = 0; __s < __num_sub_groups_per_work_group; ++__s)
        {
            __counts[__s] = __slm_subgroup_hists[__slm_idx + __s * __bin_count];
        }
        _ONEDPL_PRAGMA_UNROLL
        for (std::uint32_t __s = 0; __s < __num_sub_groups_per_work_group; ++__s)
        {
            __item_grf_hist_summary += __counts[__s];
            __slm_subgroup_hists[__slm_idx + __s * __bin_count] = __item_grf_hist_summary;
        }
        return __item_grf_hist_summary;
    }

    template <std::uint32_t __bin_process_width>
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
            __ref.store(__item_bin_count | __hist_updated_mask);
        }
        else
        {
            // WG0 is a special case: it also retrieves the total global histogram and adds it to its local histogram
            // This global histogram will be propagated to other work-groups through a chained scan at stage 2
            _GlobOffsetT __global_hist = __p_global_hist[__hist_idx] & __global_offset_mask;
            _GlobOffsetT __after_group_hist_sum = __global_hist + __item_bin_count;
            _GlobalAtomicT __ref(__p_this_group_hist[__hist_idx]);
            __ref.store(__after_group_hist_sum | __hist_updated_mask | __global_accumulated_mask);
            // Copy the global histogram to local memory to share with other work-items
            __slm_global_incoming[__hist_idx] = __global_hist;
        }
    }

    template <std::uint32_t __bin_process_width, std::uint32_t __bin_summary_sub_group_size>
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
            } while ((__prev_group_hist & __hist_updated_mask) == 0);
            __prev_group_hist_sum += __is_not_accumulated ? __prev_group_hist : 0;
            __is_not_accumulated = (__prev_group_hist_sum & __global_accumulated_mask) == 0;
            __p_lookback_hist -= __bin_count;
        } while (sycl::any_of_group(__sub_group, __is_not_accumulated));

        __prev_group_hist_sum &= __global_offset_mask;
        _GlobOffsetT __after_group_hist_sum = __prev_group_hist_sum + __item_bin_count;
        _LocIdxT __bin_idx = __sub_group_group_id * __bin_process_width + __sub_group_local_id;

        // 2.2. Write the histogram scanned across work-group, updated with the current work-group data
        _GlobalAtomicT __ref(__p_this_group_hist[__bin_idx]);
        __ref.store(__after_group_hist_sum | __hist_updated_mask | __global_accumulated_mask);

        // 2.3. Save the scanned histogram from previous work-groups locally
        __slm_global_incoming[__bin_idx] = __prev_group_hist_sum;
    }

    void inline __propagate_ranks_across_sub_groups(_LocOffsetT (&__ranks)[__data_per_work_item],
                                                    const _KeyT (&__keys)[__data_per_work_item],
                                                    _LocOffsetT* __slm_subgroup_hists, _LocOffsetT* __slm_group_hist,
                                                    std::uint32_t __sub_group_id) const
    {
        // update ranks to reflect sub-group offsets in and across bins
        _ONEDPL_PRAGMA_UNROLL
        for (std::uint32_t __i = 0; __i < __data_per_work_item; ++__i)
        {
            _LocOffsetT __bin = __key_bin_after_scans(__keys[__i]);
            _LocOffsetT __offset_in_bin =
                (__sub_group_id == 0) ? 0 : __slm_subgroup_hists[(__sub_group_id - 1) * __bin_count + __bin];
            _LocOffsetT __offset_across_bins = __slm_group_hist[__bin];
            __ranks[__i] += __offset_in_bin + __offset_across_bins;
        }
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
            _LocIdxT __bin = __get_bucket_scalar<__mask>(
                oneapi::dpl::__internal::__order_preserving_cast<__is_ascending>(__key), __stage * __radix_bits);
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
        sycl::group __group = __idx.get_group();
        sycl::sub_group __sub_group = __idx.get_sub_group();
        const std::uint32_t __sg_id = __sub_group.get_group_linear_id();
        const std::uint32_t __sg_local_id = __sub_group.get_local_id();

        const std::uint32_t __sub_group_slm_offset = __sg_id * __bin_count;

        // Tiles are handed out in increasing order through a counter, so the tiles in flight stay consecutive
        // even when work-groups progress at different speeds.
        using _CounterAtomicT = sycl::atomic_ref<_GlobOffsetT, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                                 sycl::access::address_space::global_space>;
        _CounterAtomicT __tile_counter(__p_group_hists[__num_tiles * __bin_count]);
        std::uint32_t* __slm_next_tile = reinterpret_cast<std::uint32_t*>(
            __slm_accessor.get_multi_ptr<sycl::access::decorated::no>().get() + __calc_slm_alloc() -
            sizeof(std::uint32_t));
        if (__idx.get_local_linear_id() == 0)
            *__slm_next_tile = __tile_counter.fetch_add(1);
        sycl::group_barrier(__group);
        for (std::uint32_t __tile_id = *__slm_next_tile; __tile_id < __num_tiles; __tile_id = *__slm_next_tile)
        {
            auto __values_pack = __make_key_value_pack<__data_per_work_item, _KeyT, _ValT>();
            _LocOffsetT __ranks[__data_per_work_item];

            __load_pack(__values_pack, __tile_id, __sg_id, __sg_local_id);
            if (__clear_input)
            {
                __clear_pack(__tile_id, __sg_id, __sg_local_id);
            }

            // Get raw SLM pointer and create typed pointers for different regions using helper functions
            unsigned char* __slm_raw = __slm_accessor.get_multi_ptr<sycl::access::decorated::no>().get();
            _LocOffsetT* __slm_subgroup_hists = reinterpret_cast<_LocOffsetT*>(__slm_raw);
            _LocOffsetT* __slm_group_hist = reinterpret_cast<_LocOffsetT*>(__slm_raw + __get_slm_group_hist_offset());
            _GlobOffsetT* __slm_global_incoming =
                reinterpret_cast<_GlobOffsetT*>(__slm_raw + __get_slm_global_incoming_offset());

            __rank_local(__idx, __sub_group, __ranks, __values_pack.__keys, __slm_subgroup_hists,
                         __sub_group_slm_offset, __sg_local_id);
            const _LocOffsetT __item_bin_count = __rank_global(__idx, __sub_group, __tile_id, __sg_id, __sg_local_id,
                                                               __slm_subgroup_hists, __slm_group_hist,
                                                               __slm_global_incoming);

            // Reorder the keys into SLM before the lookback: it only needs tile-local offsets. The sub-groups that do
            // not take part in the lookback then do useful work while it runs, and the lookback starts a little later,
            // when more predecessors have published their inclusive histograms.
            __propagate_ranks_across_sub_groups(__ranks, __values_pack.__keys, __slm_subgroup_hists, __slm_group_hist,
                                                __sg_id);
            // The reorder space overlaps with the sub-group histogram region
            sycl::group_barrier(__group);
            _KeyT* __slm_keys = reinterpret_cast<_KeyT*>(__slm_raw);
            _ValT* __slm_vals = nullptr;
            if constexpr (__has_values)
            {
                __slm_vals =
                    reinterpret_cast<_ValT*>(__slm_raw + __work_group_size * __data_per_work_item * sizeof(_KeyT));
            }
            _ONEDPL_PRAGMA_UNROLL
            for (std::uint32_t __i = 0; __i < __data_per_work_item; ++__i)
            {
                __slm_keys[__ranks[__i]] = __values_pack.__keys[__i];
                if constexpr (__has_values)
                {
                    __slm_vals[__ranks[__i]] = __values_pack.__vals[__i];
                }
            }

            __rank_global_lookback(__idx, __sub_group, __tile_id, __sg_id, __sg_local_id, __item_bin_count,
                                   __slm_group_hist, __slm_global_incoming);
            // Request the next tile once this tile's lookback is done: the atomic's latency overlaps with the scatter
            // while tiles are still handed out roughly in the order in which their predecessors finish
            std::uint32_t __next_tile_id = 0;
            if (__idx.get_local_linear_id() == 0)
                __next_tile_id = __tile_counter.fetch_add(1);
            // Reordered keys and global offsets are visible to the work-group
            sycl::group_barrier(__group);

            __reorder_slm_to_glob(__idx, __values_pack, __sg_id, __sg_local_id, __slm_global_incoming, __slm_keys,
                                  __slm_vals);

            if (__idx.get_local_linear_id() == 0)
                *__slm_next_tile = __next_tile_id;
            sycl::group_barrier(__group);
            // Make sure our atomic updates are pushed to other groups
            sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::device);
        }
    }
};

} // namespace oneapi::dpl::experimental::kt::gpu::__impl

#endif // _ONEDPL_KT_SYCL_RADIX_SORT_KERNELS_H
