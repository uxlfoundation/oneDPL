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

#include "radix_sort_utils.h"
#include "esimd_radix_sort_utils.h"

namespace oneapi::dpl::experimental::kt::gpu::__impl
{

//-----------------------------------------------------------------------------
// SYCL global histogram kernel implementation
//-----------------------------------------------------------------------------
template <bool __is_ascending, std::uint8_t __radix_bits, std::uint32_t __num_histograms,
          std::uint32_t __hist_work_group_count, std::uint16_t __hist_work_group_size, std::uint32_t __sub_group_size,
          typename _KeysRng>
void
__sycl_global_histogram(sycl::nd_item<1> __idx, std::size_t __n, const _KeysRng& __keys_rng, std::uint32_t* __slm,
                        std::uint32_t* __p_global_offset)
{
    using _KeyT = oneapi::dpl::__internal::__value_t<_KeysRng>;
    using _BinT = std::uint16_t;
    using _GlobalHistT = std::uint32_t;

    constexpr std::uint32_t __hist_num_sub_groups = __hist_work_group_size / __sub_group_size;
    constexpr std::uint32_t __bin_count = 1 << __radix_bits;
    constexpr std::uint32_t __bit_count = sizeof(_KeyT) * 8;
    constexpr std::uint32_t __stage_count = oneapi::dpl::__internal::__dpl_ceiling_div(__bit_count, __radix_bits);
    constexpr std::uint32_t __hist_data_per_sub_group = 128;
    constexpr std::uint32_t __hist_data_per_work_item = __hist_data_per_sub_group / __sub_group_size;
    constexpr std::uint32_t __device_wide_step =
        __hist_work_group_count * __hist_work_group_size * __hist_data_per_work_item;
    constexpr std::uint32_t __hist_buffer_size = __stage_count * __bin_count;

    const std::uint32_t __local_id = __idx.get_local_linear_id();
    const std::uint32_t __group_id = __idx.get_group_linear_id();
    const std::uint32_t __sub_group_id = __idx.get_sub_group().get_group_linear_id();
    const std::uint32_t __sub_group_local_id = __idx.get_sub_group().get_local_linear_id();

#ifdef DEBUG_SYCL_KT
    const std::uint32_t __num_groups = __idx.get_group_range(0);
    if (__local_id == 0 && __group_id == __num_groups - 1)
    {
        sycl::ext::oneapi::experimental::printf("[HISTOGRAM WG %u] Starting global histogram kernel\n", __group_id);
    }
#endif

    std::uint32_t __sub_group_start = (__group_id * __hist_num_sub_groups + __sub_group_id) * __hist_data_per_sub_group;

    // 0. Early exit - important for small inputs as we intentionally oversubscribe the hardware
    if ((__sub_group_start - __sub_group_id * __hist_data_per_sub_group) >= __n)
        return;

    // 1. Initialize group-local histograms in SLM
    for (std::uint32_t __i = __local_id; __i < __hist_buffer_size; __i += __hist_work_group_size)
    {
        _ONEDPL_PRAGMA_UNROLL
        for (std::uint32_t __j = 0; __j < __num_histograms; ++__j)
        {
            __slm[__i * __num_histograms + __j] = 0;
        }
    }

    sycl::group_barrier(__idx.get_group());

#ifdef DEBUG_SYCL_KT
    if (__local_id == 0 && __group_id == __idx.get_group_range(0) - 1)
    {
        sycl::ext::oneapi::experimental::printf("[HISTOGRAM WG %u] SLM initialized\n", __group_id);
    }
#endif

    for (std::uint32_t __wi_offset = __sub_group_start + __sub_group_local_id; __wi_offset < __n;
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

#ifdef DEBUG_SYCL_KT
        if (__local_id == 0 && __group_id == __idx.get_group_range(0) - 1)
        {
            sycl::ext::oneapi::experimental::printf("[histogram wg %u] keys read at offset %u\n", __group_id,
                                                    __wi_offset);
        }
#endif

        // 3. Accumulate histogram to SLM
        // SLM uses a blocked layout where each bin contains _NumHistograms sub-bins that are used to reduce contention
        // during atomic accumulation.
        _ONEDPL_PRAGMA_UNROLL
        for (std::uint32_t __stage = 0; __stage < __stage_count; ++__stage)
        {
            constexpr _BinT __mask = __bin_count - 1;
            _ONEDPL_PRAGMA_UNROLL
            for (std::uint32_t __i = 0; __i < __hist_data_per_work_item; ++__i)
            {
                _BinT __bucket = __get_bucket_scalar<__mask>(
                    __order_preserving_cast_scalar<__is_ascending>(__keys[__i]), __stage * __radix_bits);
                _GlobalHistT __bin = __stage * __bin_count + __bucket;
                using _SLMAtomicRef =
                    sycl::atomic_ref<_GlobalHistT, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                     sycl::access::address_space::local_space>;
                // Use sub group local id to randomize sub-bin selection for histogram accumulation
                auto __slm_hist_idx = __sub_group_local_id % __num_histograms;
                _SLMAtomicRef __slm_ref(__slm[__bin * __num_histograms + __slm_hist_idx]);
                __slm_ref.fetch_add(1);
            }
        }
    }

    sycl::group_barrier(__idx.get_group());

#ifdef DEBUG_SYCL_KT
    if (__local_id == 0 && __group_id == __idx.get_group_range(0) - 1)
    {
        sycl::ext::oneapi::experimental::printf("[HISTOGRAM WG %u] Histogram accumulated to SLM\n", __group_id);
    }
#endif

    // 4. Reduce group-local histograms from SLM into global histograms in global memory
    for (std::uint32_t __i = __local_id; __i < __hist_buffer_size; __i += __hist_work_group_size)
    {
        using _AtomicRef = sycl::atomic_ref<_GlobalHistT, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                            sycl::access::address_space::global_space>;
        _GlobalHistT __reduced_bincount = 0;
        // Blocked layout enables load vectorization from SLM
        _ONEDPL_PRAGMA_UNROLL
        for (std::uint32_t __j = 0; __j < __num_histograms; ++__j)
        {
            __reduced_bincount += __slm[__i * __num_histograms + __j];
        }
        _AtomicRef __global_hist_ref(__p_global_offset[__i]);
        __global_hist_ref.fetch_add(__reduced_bincount);
    }

#ifdef DEBUG_SYCL_KT
    if (__local_id == 0 && __group_id == __idx.get_group_range(0) - 1)
    {
        sycl::ext::oneapi::experimental::printf("[HISTOGRAM WG %u] Histogram kernel complete\n", __group_id);
    }
#endif
}

template <typename _KtTag, bool __is_ascending, ::std::uint8_t __radix_bits, ::std::uint16_t __data_per_work_item,
          ::std::uint16_t __work_group_size, typename _InRngPack, typename _OutRngPack>
struct __radix_sort_onesweep_kernel;

template <bool __is_ascending, ::std::uint8_t __radix_bits, ::std::uint16_t __data_per_work_item,
          ::std::uint16_t __work_group_size, typename _InRngPack, typename _OutRngPack>
struct __radix_sort_onesweep_kernel<__sycl_tag, __is_ascending, __radix_bits, __data_per_work_item, __work_group_size, _InRngPack, _OutRngPack>
{
    using _LocOffsetT = ::std::uint32_t;
    using _GlobOffsetT = ::std::uint32_t;
    using _AtomicIdT = ::std::uint32_t;

    using _KeyT = typename _InRngPack::_KeyT;
    using _ValT = typename _InRngPack::_ValT;
    static constexpr bool __has_values = !::std::is_void_v<_ValT>;

    static constexpr ::std::uint32_t __bin_count = 1 << __radix_bits;

    template <typename _T, ::std::uint16_t _N>
    using _SimdT = __dpl_esimd::__ns::simd<_T, _N>;

    using _LocOffsetSimdT = _SimdT<_LocOffsetT, __data_per_work_item>;
    using _GlobOffsetSimdT = _SimdT<_GlobOffsetT, __data_per_work_item>;
    using _LocHistT = _SimdT<_LocOffsetT, __bin_count>;
    using _GlobHistT = _SimdT<_GlobOffsetT, __bin_count>;

    static constexpr ::std::uint32_t __sub_group_size = 32;
    static constexpr ::std::uint32_t __num_sub_groups_per_work_group = __work_group_size / __sub_group_size;
    static constexpr ::std::uint32_t __data_per_sub_group = __data_per_work_item * __sub_group_size;

    static constexpr ::std::uint32_t __bit_count = sizeof(_KeyT) * 8;
    static constexpr _LocOffsetT __mask = __bin_count - 1;
    static constexpr ::std::uint32_t __hist_stride = __bin_count * sizeof(_LocOffsetT);
    static constexpr ::std::uint32_t __work_item_all_hists_size = __num_sub_groups_per_work_group * __hist_stride;
    static constexpr ::std::uint32_t __group_hist_size = __hist_stride; // _LocOffsetT
    static constexpr ::std::uint32_t __global_hist_size = __bin_count * sizeof(_GlobOffsetT);

    static constexpr ::std::uint32_t
    __calc_reorder_slm_size()
    {
        if constexpr (__has_values)
            return __work_group_size * __data_per_work_item * (sizeof(_KeyT) + sizeof(_ValT));
        else
            return __work_group_size * __data_per_work_item * sizeof(_KeyT);
    }

    static constexpr ::std::uint32_t
    __calc_slm_alloc()
    {
        // SLM usage:
        // 1. Getting offsets:
        //      1.1 Scan histograms for each work-item: __work_item_all_hists_size
        //      1.2 Scan group histogram: __group_hist_size
        //      1.3 Accumulate group histogram from previous groups: __global_hist_size
        // 2. Reorder keys in SLM:
        //      2.1 Reorder key-value pairs: __reorder_size (overlaps with histogram space)
        //      2.2 Place global offsets into SLM for lookup: __global_hist_size (after all histograms)
        //      During reorder, we need the old histograms + global_fix simultaneously
        constexpr ::std::uint32_t __reorder_size = __calc_reorder_slm_size();
        constexpr ::std::uint32_t __offset_calc_substage_slm =
            __work_item_all_hists_size + __group_hist_size + __global_hist_size;
        constexpr ::std::uint32_t __reorder_substage_slm =
            __offset_calc_substage_slm + __global_hist_size; // Need space for old histograms + global_fix

        constexpr ::std::uint32_t __slm_size = ::std::max(__offset_calc_substage_slm, __reorder_substage_slm);
        // Workaround: Align SLM allocation at 2048 byte border to avoid internal compiler error.
        // The error happens when allocating 65 * 1024 bytes, when e.g. T=int, DataPerWorkItem=256, WorkGroupSize=64
        // TODO: use __slm_size once the issue with SLM allocation has been fixed
        return oneapi::dpl::__internal::__dpl_ceiling_div(__slm_size, 2048) * 2048;
    }

    const ::std::uint32_t __n;
    const ::std::uint32_t __stage;
    _GlobOffsetT* __p_global_hist;
    _GlobOffsetT* __p_group_hists;
    _AtomicIdT* __p_atomic_id;
    _InRngPack __in_pack;
    _OutRngPack __out_pack;
    sycl::local_accessor<unsigned char, 1> __slm_accessor;

    __radix_sort_onesweep_kernel(::std::uint32_t __n, ::std::uint32_t __stage, _GlobOffsetT* __p_global_hist,
                                 _GlobOffsetT* __p_group_hists, _AtomicIdT* __p_atomic_id, const _InRngPack& __in_pack,
                                 const _OutRngPack& __out_pack, sycl::local_accessor<unsigned char, 1> __slm_acc)
        : __n(__n), __stage(__stage), __p_global_hist(__p_global_hist), __p_group_hists(__p_group_hists),
          __p_atomic_id(__p_atomic_id), __in_pack(__in_pack), __out_pack(__out_pack), __slm_accessor(__slm_acc)
    {
    }

    template <typename _KVPack>
    inline auto
    __load_pack(_KVPack& __pack, std::uint32_t __wg_id, std::uint32_t __sg_id, std::uint32_t __sg_local_id) const
    {
        const _GlobOffsetT __offset = __data_per_sub_group * (__wg_id * __num_sub_groups_per_work_group + __sg_id);
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
        //static_assert(__data_per_work_item % __data_per_step == 0);
        bool __is_full_block = (__glob_offset + __data_per_sub_group) <= __n;
        auto __offset = __glob_offset + __local_offset;
        if (__is_full_block)
        {
            _ONEDPL_PRAGMA_UNROLL
            for (::std::uint32_t __i = 0; __i < __data_per_work_item; ++__i)
            {
                __elements[__i] = __in_seq[__offset + __i * __sub_group_size];
            }
        }
        else
        {
            _ONEDPL_PRAGMA_UNROLL
            for (::std::uint32_t __i = 0; __i < __data_per_work_item; ++__i)
            {
                auto __idx = __offset + __i * __sub_group_size;
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
    __match_bins(sycl::nd_item<1> __idx, std::uint32_t __bin)
    {
        // start with all bits 1
        auto __matched_bins = sycl::ext::oneapi::group_ballot(__idx.get_sub_group());
        _ONEDPL_PRAGMA_UNROLL
        for (int __i = 0; __i < __radix_bits; __i++)
        {
            auto __bit = (__bin >> __i) & 1;
            auto __sg_vote = sycl::ext::oneapi::group_ballot(__idx.get_sub_group(), static_cast<bool>(__bit));
            // If we vote yes, then we want to set all bits that also voted yes. If no, then we want to
            // zero out the bits that said yes as they don't match and preserve others as we have no info on these.
            __matched_bins &= __bit ? __sg_vote : ~__sg_vote;
        }
        std::uint32_t __result = 0;
        __matched_bins.extract_bits(__result);
        return __result;
    }

    inline auto
    __rank_local(sycl::nd_item<1> __idx, _LocOffsetT __ranks[__data_per_work_item],
                 _LocOffsetT __bins[__data_per_work_item], _LocOffsetT* __slm_subgroup_hists,
                 std::uint32_t __sub_group_slm_offset) const
    {
        std::uint32_t __sub_group_local_id = __idx.get_sub_group().get_local_id();
        _LocOffsetT* __slm_offset = __slm_subgroup_hists + __sub_group_slm_offset;

        for (std::uint32_t __i = __idx.get_sub_group().get_local_id(); __i < __bin_count; __i += __sub_group_size)
        {
            __slm_offset[__i] = 0;
        }
        // TODO: sub-group barrier ? maybe not for simd architectures
        // sub-group barrier or no?

        //_ScanSimdT __remove_right_lanes, __lane_id(0, 1);
        constexpr std::uint32_t __sub_group_full_bitmask = 0x7fffffff;
        static_assert(__sub_group_size == 32);
        // lower bits than my current will be set meaning we only preserve left lanes
        std::uint32_t __remove_right_lanes = __sub_group_full_bitmask >> (__sub_group_size - 1 - __sub_group_local_id);

        //static_assert(__data_per_work_item % __bins_per_step == 0);
        _ONEDPL_PRAGMA_UNROLL
        for (std::uint32_t __i = 0; __i < __data_per_work_item; ++__i)
        {
            _LocOffsetT __bin = __bins[__i];
            std::uint32_t __matched_bins = __match_bins(__idx, __bin);
            std::uint32_t __pre_rank = __slm_offset[__bin];
            auto __matched_left_lanes = __matched_bins & __remove_right_lanes;
            std::uint32_t __this_round_rank = sycl::popcount(__matched_left_lanes);
            std::uint32_t __this_round_count = sycl::popcount(__matched_bins);
            auto __rank_after = __pre_rank + __this_round_rank;
            auto __is_leader = __this_round_rank == __this_round_count - 1;
            if (__is_leader)
            {
                __slm_offset[__bin] = __rank_after + 1;
            }
            __ranks[__i] = __rank_after;
        }
        sycl::group_barrier(__idx.get_group());
    }

    inline void
    __rank_global(sycl::nd_item<1> __idx, std::uint32_t __wg_id, _LocOffsetT* __slm_subgroup_hists,
                  _LocOffsetT* __slm_group_hist, _GlobOffsetT* __slm_global_incoming) const
    {
        // SLM layout (in bytes):
        // [0 ... work_item_all_hists_size): Sub-group histograms (_LocOffsetT)
        // [work_item_all_hists_size ... work_item_all_hists_size + group_hist_size): Group histogram (_LocOffsetT)
        // [work_item_all_hists_size + group_hist_size ... ): Global incoming histogram (_GlobOffsetT)

        constexpr ::std::uint32_t __global_accumulated = 0x40000000;
        constexpr ::std::uint32_t __hist_updated = 0x80000000;
        constexpr ::std::uint32_t __global_offset_mask = 0x3fffffff;

        _GlobOffsetT* __p_this_group_hist = __p_group_hists + __bin_count * __wg_id;
        _GlobOffsetT* __p_prev_group_hist = __p_this_group_hist - __bin_count;

        // This is important so that we can evenly partition the radix bits across a number of sub-groups
        // without masking lanes. Radix bits is always a power of two, so this requirement essentially just
        // requires radix_bits >= 5 for sub-group size of 32.
        static_assert(__bin_count % __sub_group_size == 0);

        constexpr ::std::uint32_t __bin_summary_sub_group_size = __bin_count / __sub_group_size;
        constexpr ::std::uint32_t __bin_width = __sub_group_size;

        auto __sub_group_id = __idx.get_sub_group().get_group_linear_id();
        auto __sub_group_local_id = __idx.get_sub_group().get_local_linear_id();

        // 1. Vector scan of histograms previously accumulated by each work-item
        // update slm instead of grf summary due to perf issues with grf histogram

        // TODO: this single element array is a temporary workaround for sub group scan requiring an array
        _LocOffsetT __item_grf_hist_summary_arr[1] = {0};
        _LocOffsetT& __item_grf_hist_summary = __item_grf_hist_summary_arr[0];
        _LocOffsetT __item_bin_count;
        if (__sub_group_id < __bin_summary_sub_group_size)
        {
            // 1.1. Vector scan of the same bins across different histograms.
            std::uint32_t __slm_bin_hist_summary_offset = __sub_group_id * __bin_width;

            for (::std::uint32_t __s = 0; __s < __num_sub_groups_per_work_group;
                 __s++, __slm_bin_hist_summary_offset += __bin_count)
            {
                auto __slm_idx = __slm_bin_hist_summary_offset + __sub_group_local_id;
                __item_grf_hist_summary += __slm_subgroup_hists[__slm_idx];
                __slm_subgroup_hists[__slm_idx] = __item_grf_hist_summary;
            }
            __item_bin_count = __item_grf_hist_summary;

            // 1.2. Vector scan of different bins inside one histogram, the final one for the whole work-group.
            // Only "__bin_width" pieces of the histogram are scanned at this stage.
            // This histogram will be further used for calculation of offsets of keys already reordered in SLM,
            // it does not participate in sycnhronization between work-groups.
            __sub_group_scan<__sub_group_size, 1>(__idx.get_sub_group(), __item_grf_hist_summary_arr, std::plus<>{},
                                                  __bin_width);

            auto __write_idx = __sub_group_id * __bin_width + __sub_group_local_id;
            __slm_group_hist[__write_idx] = __item_grf_hist_summary;

            // 1.3. Copy the histogram at the region designated for synchronization between work-groups.
            // Write the histogram to global memory, bypassing caches, to ensure cross-work-group visibility.
            // TODO: write to L2 if only one stack is used for better performance
            if (__wg_id != 0)
            {
                // Copy the histogram, local to this WG
                using _GlobalAtomicT = sycl::atomic_ref<_GlobOffsetT, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                                         sycl::access::address_space::global_space>;
                _GlobalAtomicT __ref(__p_this_group_hist[__sub_group_id * __bin_width + __sub_group_local_id]);
                __ref.store(__item_bin_count | __hist_updated);
            }
            else
            {
                // WG0 is a special case: it also retrieves the total global histogram and adds it to its local histogram
                // This global histogram will be propagated to other work-groups through a chained scan at stage 2
                using _GlobalAtomicT =
                    sycl::atomic_ref<_GlobOffsetT, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                     sycl::access::address_space::global_space>;

                auto __hist_idx = __sub_group_id * __bin_width + __sub_group_local_id;
                _GlobOffsetT __global_hist = __p_global_hist[__hist_idx] & __global_offset_mask;
                _GlobOffsetT __after_group_hist_sum = __global_hist + __item_bin_count;
                _GlobalAtomicT __ref(__p_this_group_hist[__hist_idx]);
                __ref.store(__after_group_hist_sum | __hist_updated | __global_accumulated);
                // Copy the global histogram to local memory to share with other work-items
                __slm_global_incoming[__hist_idx] = __global_hist;
            }
        }
        sycl::group_barrier(__idx.get_group());

        auto __sub_group = __idx.get_sub_group();
        auto __sub_group_group_id = __sub_group.get_group_linear_id();

        // 1.4 One work-item finalizes scan performed at stage 1.2
        // by propagating prefixes accumulated after scanning individual "__bin_width" pieces.
        if (__sub_group_group_id == __bin_summary_sub_group_size + 1)
        {
            __sub_group_cross_segment_exclusive_scan<__bin_width, __bin_summary_sub_group_size, __sub_group_size>(
                __sub_group, __slm_group_hist);
        }

        sycl::group_barrier(__idx.get_group());

        // 2. Chained scan. Synchronization between work-groups.
        if (__sub_group_group_id < __bin_summary_sub_group_size && __wg_id != 0)
        {
            using _GlobalAtomicT = sycl::atomic_ref<_GlobOffsetT, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                                        sycl::access::address_space::global_space>;
            // 2.1. Read the histograms scanned across work-groups
            _GlobOffsetT __prev_group_hist_sum = 0;
            _GlobOffsetT __prev_group_hist;
            bool __is_not_accumulated = true;
            do
            {
                auto __idx = __sub_group_group_id * __bin_width + __sub_group_local_id;
                _GlobalAtomicT __ref(__p_prev_group_hist[__idx]);
                do
                {
                    __prev_group_hist = __ref.load();
                } while ((__prev_group_hist & __hist_updated) == 0);
                __prev_group_hist_sum += __is_not_accumulated ? __prev_group_hist : 0;
                __is_not_accumulated = (__prev_group_hist_sum & __global_accumulated) == 0;
                __p_prev_group_hist -= __bin_count;
            } while (sycl::any_of_group(__sub_group, __is_not_accumulated));

            _GlobOffsetT __after_group_hist_sum = 0;
            __prev_group_hist_sum &= __global_offset_mask;
            __after_group_hist_sum = __prev_group_hist_sum + __item_bin_count;
            auto __idx = __sub_group_group_id * __bin_width + __sub_group_local_id;
            // 2.2. Write the histogram scanned across work-group, updated with the current work-group data
            _GlobalAtomicT __ref(__p_this_group_hist[__idx]);
            __ref.store(__after_group_hist_sum | __hist_updated | __global_accumulated);
            // 2.3. Save the scanned histogram from previous work-groups locally
            __slm_global_incoming[__idx] = __prev_group_hist_sum;
        }
        sycl::group_barrier(__idx.get_group());
    }

    template <typename _KVPack>
    void inline __reorder_reg_to_slm(sycl::nd_item<1> __idx, const _KVPack& __pack,
                                     _LocOffsetT (&__ranks)[__data_per_work_item],
                                     const _LocOffsetT (&__bins)[__data_per_work_item],
                                     _LocOffsetT* __slm_subgroup_hists, _LocOffsetT* __slm_group_hist,
                                     _GlobOffsetT* __slm_global_incoming, _GlobOffsetT* __slm_global_fix,
                                     _KeyT* __slm_keys, _ValT* __slm_vals) const
    {
        auto __sub_group_id = __idx.get_sub_group().get_group_linear_id();
        const auto __wg_size = __idx.get_local_range(0);

        // 1. update ranks to reflect sub-group offsets in and across bins
        _ONEDPL_PRAGMA_UNROLL
        for (std::uint32_t __i = 0; __i < __data_per_work_item; ++__i)
        {
            auto __bin = __bins[__i];
            auto __group_incoming = __slm_group_hist[__bin];
            auto __offset_in_bin =
                (__sub_group_id == 0) ? 0 : __slm_subgroup_hists[(__sub_group_id - 1) * __bin_count + __bin];
            auto __offset_across_bins = __group_incoming;
            __ranks[__i] += __offset_in_bin + __offset_across_bins;
        }

        // 2. compute __global_fix
        for (std::uint32_t __i = __idx.get_local_id(); __i < __bin_count; __i += __wg_size)
        {
            __slm_global_fix[__i] = __slm_global_incoming[__i] - static_cast<_GlobOffsetT>(__slm_group_hist[__i]);
        }
        sycl::group_barrier(__idx.get_group());

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
        sycl::group_barrier(__idx.get_group());
    }

    template <typename _KVPack>
    void inline __reorder_slm_to_glob(sycl::nd_item<1> __idx, _KVPack& __pack, _GlobOffsetT* __slm_global_fix,
                                      _KeyT* __slm_keys, _ValT* __slm_vals) const
    {
        auto __sub_group_id = __idx.get_sub_group().get_group_linear_id();
        auto __sub_group_local_id = __idx.get_sub_group().get_local_linear_id();

        const _GlobOffsetT __keys_slm_offset = __data_per_sub_group * __sub_group_id;

        _ONEDPL_PRAGMA_UNROLL
        for (std::uint32_t __i = 0; __i < __data_per_work_item; ++__i)
        {
            auto __slm_idx = __keys_slm_offset + __i * __sub_group_size + __sub_group_local_id;
            auto __key = __slm_keys[__slm_idx];
            auto __bin = __get_bucket_scalar<__mask>(__order_preserving_cast_scalar<__is_ascending>(__key),
                                                     __stage * __radix_bits);
            auto __global_fix = __slm_global_fix[__bin];
            auto __out_idx = __global_fix + __slm_idx;

            if (__out_idx < __n)
                __out_pack.__keys_rng()[__out_idx] = __key;
            if constexpr (__has_values)
            {
                auto __val = __slm_vals[__slm_idx];
                __out_pack.__vals_rng()[__out_idx] = __val;
            }
        }
    }

    [[sycl::reqd_sub_group_size(__sub_group_size)]] void
    operator()(sycl::nd_item<1> __idx) const
    {
        const ::std::uint32_t __local_tid = __idx.get_local_linear_id();
        const ::std::uint32_t __wg_size = __idx.get_local_range(0);
        const ::std::uint32_t __sg_id = __idx.get_sub_group().get_group_linear_id();
        const ::std::uint32_t __sg_local_id = __idx.get_sub_group().get_local_id();

        const ::std::uint32_t __num_wgs = __idx.get_group_range(0);
        using _AtomicRefT = sycl::atomic_ref<_AtomicIdT, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                             sycl::access::address_space::global_space>;
        _AtomicRefT __atomic_id_ref(*__p_atomic_id);
        std::uint32_t __wg_id = 0;
        if (__idx.get_local_linear_id() == 0)
        {
            // Modulo num work-groups because onesweep gets invoked multiple times and we do not want an extra memset between
            // invocations.
            __wg_id = __atomic_id_ref.fetch_add(1) % __num_wgs;
        }
        __wg_id = sycl::group_broadcast(__idx.get_group(), __wg_id);

        // TODO: Right now storing a full sub-group histogram contiguously in SLM. Consider evaluating
        // approach where each sub-group's bins are interleaved
        const ::std::uint32_t __sub_group_slm_offset = __sg_id * __bin_count;

        auto __values_pack = __make_key_value_pack<__data_per_work_item, _KeyT, _ValT>();
        _LocOffsetT __bins[__data_per_work_item];
        _LocOffsetT __ranks[__data_per_work_item];

        __load_pack(__values_pack, __wg_id, __sg_id, __sg_local_id);

        _ONEDPL_PRAGMA_UNROLL
        for (std::uint32_t __i = 0; __i < __data_per_work_item; ++__i)
        {
            const auto __ordered = __order_preserving_cast_scalar<__is_ascending>(__values_pack.__keys[__i]);
            __bins[__i] = __get_bucket_scalar<__mask>(__ordered, __stage * __radix_bits);
        }

        // Get raw SLM pointer and create typed pointers for different regions
        unsigned char* __slm_raw = __slm_accessor.get_multi_ptr<sycl::access::decorated::no>().get();
        _LocOffsetT* __slm_subgroup_hists = reinterpret_cast<_LocOffsetT*>(__slm_raw);
        _LocOffsetT* __slm_group_hist = reinterpret_cast<_LocOffsetT*>(__slm_raw + __work_item_all_hists_size);
        _GlobOffsetT* __slm_global_incoming =
            reinterpret_cast<_GlobOffsetT*>(__slm_raw + __work_item_all_hists_size + __group_hist_size);

        __rank_local(__idx, __ranks, __bins, __slm_subgroup_hists, __sub_group_slm_offset);
        __rank_global(__idx, __wg_id, __slm_subgroup_hists, __slm_group_hist, __slm_global_incoming);

        // For reorder phase, reinterpret SLM as key/value storage and global_fix. This probably violates strict aliasing
        _KeyT* __slm_keys = reinterpret_cast<_KeyT*>(__slm_raw);
        _ValT* __slm_vals = nullptr;
        if constexpr (__has_values)
        {
            __slm_vals = reinterpret_cast<_ValT*>(__slm_raw + __wg_size * __data_per_work_item * sizeof(_KeyT));
        }
        _GlobOffsetT* __slm_global_fix = reinterpret_cast<_GlobOffsetT*>(__slm_raw + __work_item_all_hists_size +
                                                                         __group_hist_size + __global_hist_size);

        __reorder_reg_to_slm(__idx, __values_pack, __ranks, __bins, __slm_subgroup_hists, __slm_group_hist,
                             __slm_global_incoming, __slm_global_fix, __slm_keys, __slm_vals);

        __reorder_slm_to_glob(__idx, __values_pack, __slm_global_fix, __slm_keys, __slm_vals);
    }
};


} // namespace oneapi::dpl::experimental::kt::gpu::__impl

#endif // _ONEDPL_KT_SYCL_RADIX_SORT_KERNELS_H
