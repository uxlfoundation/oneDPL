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
}

template <typename _KtTag, bool __is_ascending, ::std::uint8_t __radix_bits, ::std::uint16_t __data_per_work_item,
          ::std::uint16_t __work_group_size, typename _InRngPack, typename _OutRngPack>
struct __radix_sort_onesweep_kernel;

template <bool __is_ascending, ::std::uint8_t __radix_bits, ::std::uint16_t __data_per_work_item,
          ::std::uint16_t __work_group_size, typename _InRngPack, typename _OutRngPack>
struct __radix_sort_onesweep_kernel<__sycl_tag, __is_ascending, __radix_bits, __data_per_work_item, __work_group_size, _InRngPack, _OutRngPack>
{
    using _LocOffsetT = ::std::uint16_t;
    using _GlobOffsetT = ::std::uint32_t;

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
    static constexpr ::std::uint32_t __work_item_all_hists_size = __work_group_size * __hist_stride;
    static constexpr ::std::uint32_t __group_hist_size = __hist_stride;
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
        //      2.1 Place global offsets into SLM for lookup: __global_hist_size
        //      2.2 Reorder key-value pairs: __reorder_size
        constexpr ::std::uint32_t __reorder_size = __calc_reorder_slm_size();
        constexpr ::std::uint32_t __offset_calc_substage_slm =
            __work_item_all_hists_size + __group_hist_size + __global_hist_size;
        constexpr ::std::uint32_t __reorder_substage_slm = __reorder_size + __global_hist_size;

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
    _InRngPack __in_pack;
    _OutRngPack __out_pack;

    __radix_sort_onesweep_kernel(::std::uint32_t __n, ::std::uint32_t __stage, _GlobOffsetT* __p_global_hist,
                                 _GlobOffsetT* __p_group_hists, const _InRngPack& __in_pack,
                                 const _OutRngPack& __out_pack)
        : __n(__n), __stage(__stage), __p_global_hist(__p_global_hist), __p_group_hists(__p_group_hists),
          __in_pack(__in_pack), __out_pack(__out_pack)
    {
    }

    template <typename _KVPack>
    inline auto
    __load_pack(_KVPack& __pack, std::uint32_t __wg_id, std::uint32_t __sg_id, std::uint32_t __sg_local_id) const
    {
        const _GlobOffsetT __offset = __data_per_sub_group * (__wg_id * __num_sub_groups_per_work_group + __sg_id);
        __load</*__sort_identity_residual=*/true>(__pack.__keys, __rng_data(__in_pack.__keys_rng()), __offset,
                                                  __sg_local_id);
        if constexpr (__has_values)
        {
            __load</*__sort_identity_residual=*/false>(__pack.__vals, __rng_data(__in_pack.__vals_rng()), __offset,
                                                       __sg_local_id);
        }
    }

    template <bool __sort_identity_residual, typename _T, typename _InSeq>
    inline void
    __load(_T (&__elements)[__data_per_work_item], const _InSeq& __in_seq, _GlobOffsetT __glob_offset,
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
            sycl::ext::oneapi::group_ballot(__idx.get_sub_group(), static_cast<bool>(__bit));
            // If we vote yes, then we want to set all bits that also voted yes. If no, then we want to
            // zero out the bits that said yes as they don't match and preserve others as we have no info on these.
            __matched_bins &= __bit ? __mask : ~__mask;
        }
        std::uint32_t __result = 0;
        __matched_bins.extract_bits(__result);
        return __result;
    }

    inline auto
    __rank_local(sycl::nd_item<1> __idx, _LocOffsetT __ranks[__data_per_work_item],
                 _LocOffsetT __bins[__data_per_work_item], std::uint32_t* __slm,
                 std::uint32_t __sub_group_slm_offset) const
    {
        //constexpr int __bins_per_step = 32;
        //using _ScanSimdT = __dpl_esimd::__ns::simd<::std::uint32_t, __bins_per_step>;
        // TODO add
        std::uint32_t __sub_group_local_id = __idx.get_sub_group().get_local_id();
        std::uint32_t* __slm_offset = __slm + __sub_group_slm_offset;

        //__dpl_esimd::__block_store_slm<_LocOffsetT, __bin_count>(__sub_group_slm_offset, 0);
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
    __rank_global(_LocHistT& __subgroup_offset, _GlobHistT& __global_fix, ::std::uint32_t __local_tid,
                  ::std::uint32_t __wg_id) const
    {
        const ::std::uint32_t __slm_bin_hist_group_incoming = __work_group_size * __hist_stride;
        const ::std::uint32_t __slm_bin_hist_global_incoming = __slm_bin_hist_group_incoming + __hist_stride;
        constexpr ::std::uint32_t __global_accumulated = 0x40000000;
        constexpr ::std::uint32_t __hist_updated = 0x80000000;
        constexpr ::std::uint32_t __global_offset_mask = 0x3fffffff;

        _GlobOffsetT* __p_this_group_hist = __p_group_hists + __bin_count * __wg_id;
        _GlobOffsetT* __p_prev_group_hist = __p_this_group_hist - __bin_count;

        constexpr ::std::uint32_t __bin_summary_group_size = 8;
        constexpr ::std::uint32_t __bin_width = __bin_count / __bin_summary_group_size;
        static_assert(__bin_count % __bin_width == 0);

        // 1. Vector scan of histograms previously accumulated by each work-item
        __dpl_esimd::__ns::simd<_LocOffsetT, __bin_width> __thread_grf_hist_summary(0);
        if (__local_tid < __bin_summary_group_size)
        {
            // 1.1. Vector scan of the same bins across different histograms.
            ::std::uint32_t __slm_bin_hist_summary_offset = __local_tid * __bin_width * sizeof(_LocOffsetT);
            for (::std::uint32_t __s = 0; __s < __work_group_size;
                 __s++, __slm_bin_hist_summary_offset += __hist_stride)
            {
                __thread_grf_hist_summary +=
                    __dpl_esimd::__block_load_slm<_LocOffsetT, __bin_width>(__slm_bin_hist_summary_offset);
                __dpl_esimd::__block_store_slm(__slm_bin_hist_summary_offset, __thread_grf_hist_summary);
            }

            // 1.2. Vector scan of different bins inside one histogram, the final one for the whole work-group.
            // Only "__bin_width" pieces of the histogram are scanned at this stage.
            // This histogram will be further used for calculation of offsets of keys already reordered in SLM,
            // it does not participate in sycnhronization between work-groups.
            __dpl_esimd::__block_store_slm(__slm_bin_hist_group_incoming +
                                               __local_tid * __bin_width * sizeof(_LocOffsetT),
                                           __scan<_LocOffsetT, _LocOffsetT>(__thread_grf_hist_summary));

            // 1.3. Copy the histogram at the region designated for synchronization between work-groups.
            // Write the histogram to global memory, bypassing caches, to ensure cross-work-group visibility.
            // TODO: write to L2 if only one stack is used for better performance
            if (__wg_id != 0)
            {
                // Copy the histogram, local to this WG
                __dpl_esimd::__ens::lsc_block_store<_GlobOffsetT, __bin_width,
                                                    __dpl_esimd::__ens::lsc_data_size::default_size,
                                                    __dpl_esimd::__ens::cache_hint::uncached,
                                                    __dpl_esimd::__ens::cache_hint::uncached>(
                    __p_this_group_hist + __local_tid * __bin_width, __thread_grf_hist_summary | __hist_updated);
            }
            else
            {
                // WG0 is a special case: it also retrieves the total global histogram and adds it to its local histogram
                // This global histogram will be propagated to other work-groups through a chained scan at stage 2
                __dpl_esimd::__ns::simd<_GlobOffsetT, __bin_width> __global_hist =
                     __dpl_esimd::__ens::lsc_block_load<_GlobOffsetT, __bin_width>(__p_global_hist + __local_tid * __bin_width);
                __global_hist &= __global_offset_mask;
                __dpl_esimd::__ns::simd<_GlobOffsetT, __bin_width> __after_group_hist_sum =
                    __global_hist + __thread_grf_hist_summary;
                __dpl_esimd::__ens::lsc_block_store<_GlobOffsetT, __bin_width,
                                                    __dpl_esimd::__ens::lsc_data_size::default_size,
                                                    __dpl_esimd::__ens::cache_hint::uncached,
                                                    __dpl_esimd::__ens::cache_hint::uncached>(
                    __p_this_group_hist + __local_tid * __bin_width, __after_group_hist_sum | __hist_updated | __global_accumulated);
                // Copy the global histogram to local memory to share with other work-items
                __dpl_esimd::__block_store_slm<_GlobOffsetT, __bin_width>(
                    __slm_bin_hist_global_incoming + __local_tid * __bin_width * sizeof(_GlobOffsetT),
                    __global_hist);
            }
        }
        // Make sure the histogram updated at the step 1.3 is visible to other groups
        // The histogram data above is in global memory: no need to flush caches
#if _ONEDPL_ESIMD_LSC_FENCE_PRESENT
        __dpl_esimd::__ns::fence<__dpl_esimd::__ns::memory_kind::global,
                                 __dpl_esimd::__ns::fence_flush_op::none,
                                 __dpl_esimd::__ns::fence_scope::gpu>();
#else
        __dpl_esimd::__ns::fence<__dpl_esimd::__ns::fence_mask::global_coherent_fence>();
#endif
        __dpl_esimd::__ns::barrier();

        // 1.4 One work-item finalizes scan performed at stage 1.2
        // by propagating prefixes accumulated after scanning individual "__bin_width" pieces.
        if (__local_tid == __bin_summary_group_size + 1)
        {
            __dpl_esimd::__ns::simd<_LocOffsetT, __bin_count> __grf_hist_summary;
            __dpl_esimd::__ns::simd<_LocOffsetT, __bin_count + 1> __grf_hist_summary_scan;
            __grf_hist_summary = __dpl_esimd::__block_load_slm<_LocOffsetT, __bin_count>(__slm_bin_hist_group_incoming);
            __grf_hist_summary_scan[0] = 0;
            __grf_hist_summary_scan.template select<__bin_width, 1>(1) =
                __grf_hist_summary.template select<__bin_width, 1>(0);
            _ONEDPL_PRAGMA_UNROLL
            for (::std::uint32_t __i = __bin_width; __i < __bin_count; __i += __bin_width)
            {
                __grf_hist_summary_scan.template select<__bin_width, 1>(__i + 1) =
                    __grf_hist_summary.template select<__bin_width, 1>(__i) + __grf_hist_summary_scan[__i];
            }
            __dpl_esimd::__block_store_slm<_LocOffsetT, __bin_count>(
                __slm_bin_hist_group_incoming, __grf_hist_summary_scan.template select<__bin_count, 1>());
        }

        // 2. Chained scan. Synchronization between work-groups.
        else if (__local_tid < __bin_summary_group_size && __wg_id != 0)
        {
            // 2.1. Read the histograms scanned across work-groups
            __dpl_esimd::__ns::simd<_GlobOffsetT, __bin_width> __prev_group_hist_sum(0), __prev_group_hist;
            __dpl_esimd::__ns::simd_mask<__bin_width> __is_not_accumulated(1);
            do
            {
                do
                {
                    // Read the histogram from L2, bypassing L1
                    // L1 is assumed to be non-coherent, thus it is avoided to prevent reading stale data
                    __prev_group_hist = __dpl_esimd::__ens::lsc_block_load<
                        _GlobOffsetT, __bin_width, __dpl_esimd::__ens::lsc_data_size::default_size,
                        __dpl_esimd::__ens::cache_hint::uncached, __dpl_esimd::__ens::cache_hint::cached>(
                        __p_prev_group_hist + __local_tid * __bin_width);
                    // TODO: This fence is added to prevent a hang that occurs otherwise. However, this fence
                    // should not logically be needed. Consider removing once this has been further investigated.
                    // This preprocessor check is set to expire and needs to be reevaluated once the SYCL major version
                    // is upgraded to 9.
#if _ONEDPL_LIBSYCL_VERSION < 90000
#   if _ONEDPL_ESIMD_LSC_FENCE_PRESENT
                    __dpl_esimd::__ns::fence<__dpl_esimd::__ns::memory_kind::local>();
#   else
                    __dpl_esimd::__ns::fence<__dpl_esimd::__ns::fence_mask::sw_barrier>();
#   endif
#endif
                } while (((__prev_group_hist & __hist_updated) == 0).any());
                __prev_group_hist_sum.merge(__prev_group_hist_sum + __prev_group_hist, __is_not_accumulated);
                __is_not_accumulated = (__prev_group_hist_sum & __global_accumulated) == 0;
                __p_prev_group_hist -= __bin_count;
            } while (__is_not_accumulated.any());
            __prev_group_hist_sum &= __global_offset_mask;
            __dpl_esimd::__ns::simd<_GlobOffsetT, __bin_width> __after_group_hist_sum =
                __prev_group_hist_sum + __thread_grf_hist_summary;
            // 2.2. Write the histogram scanned across work-group, updated with the current work-group data
            __dpl_esimd::__block_store<_GlobOffsetT, __bin_width>(__p_this_group_hist + __local_tid * __bin_width,
                                                                  __after_group_hist_sum | __hist_updated |
                                                                  __global_accumulated);
            // 2.3. Save the scanned histogram from previous work-groups locally
            __dpl_esimd::__block_store_slm<_GlobOffsetT, __bin_width>(
                __slm_bin_hist_global_incoming + __local_tid * __bin_width * sizeof(_GlobOffsetT),
                __prev_group_hist_sum);
        }
        __dpl_esimd::__ns::barrier();

        // 3. Get total offsets for each work-item
        auto __group_incoming = __dpl_esimd::__block_load_slm<_LocOffsetT, __bin_count>(__slm_bin_hist_group_incoming);
        // 3.1. Get histogram accumullated from previous groups (together with the global one) and apply correction for keys already reordered in SLM
        // TODO: rename the variable to represent its purpose better.
        __global_fix =
            __dpl_esimd::__block_load_slm<_GlobOffsetT, __bin_count>(__slm_bin_hist_global_incoming) - __group_incoming;
        // 3.2. Get histogram with offsets for each work-item within its work-group.
        if (__local_tid > 0)
        {
            __subgroup_offset = __group_incoming + __dpl_esimd::__block_load_slm<_LocOffsetT, __bin_count>(
                                                       (__local_tid - 1) * __hist_stride);
        }
        else
        {
            __subgroup_offset = __group_incoming;
        }
        __dpl_esimd::__ns::barrier();
    }

    template <typename _SimdPack>
    void inline __reorder_reg_to_slm(const _SimdPack& __pack, const _LocOffsetSimdT& __ranks,
                                     const _LocOffsetSimdT& __bins, const _LocHistT& __subgroup_offset,
                                     ::std::uint32_t __wg_size, ::std::uint32_t __thread_slm_offset) const
    {
        __slm_lookup<_LocOffsetT> __subgroup_lookup(__thread_slm_offset);
        _LocOffsetSimdT __wg_offset =
            __ranks + __subgroup_lookup.template __lookup<__data_per_work_item>(__subgroup_offset, __bins);
        __dpl_esimd::__ns::barrier();

        _GlobOffsetSimdT __wg_offset_keys = __wg_offset * sizeof(_KeyT);
        __dpl_esimd::__vector_store<_KeyT, 1, __data_per_work_item>(__wg_offset_keys, __pack.__keys);
        if constexpr (__has_values)
        {
            _GlobOffsetSimdT __wg_offset_vals =
                __wg_size * __data_per_work_item * sizeof(_KeyT) + __wg_offset * sizeof(_ValT);
            __dpl_esimd::__vector_store<_ValT, 1, __data_per_work_item>(__wg_offset_vals, __pack.__vals);
        }
        __dpl_esimd::__ns::barrier();
    }

    template <typename _SimdPack>
    void inline __reorder_slm_to_glob(_SimdPack& __pack, const _GlobHistT& __global_fix, ::std::uint32_t __local_tid,
                                      ::std::uint32_t __wg_size) const
    {
        __slm_lookup<_GlobOffsetT> __global_fix_lookup(__calc_reorder_slm_size());
        if (__local_tid == 0)
            __global_fix_lookup.__setup(__global_fix);
        __dpl_esimd::__ns::barrier();

        ::std::uint32_t __keys_slm_offset = __local_tid * __data_per_work_item * sizeof(_KeyT);
        __pack.__keys = __dpl_esimd::__block_load_slm<_KeyT, __data_per_work_item>(__keys_slm_offset);
        if constexpr (__has_values)
        {
            ::std::uint32_t __vals_slm_offset =
                __wg_size * __data_per_work_item * sizeof(_KeyT) + __local_tid * __data_per_work_item * sizeof(_ValT);
            __pack.__vals = __dpl_esimd::__block_load_slm<_ValT, __data_per_work_item>(__vals_slm_offset);
        }
        const auto __ordered = __order_preserving_cast<__is_ascending>(__pack.__keys);
        _LocOffsetSimdT __bins = __get_bucket<__mask>(__ordered, __stage * __radix_bits);

        // This vector contains IDs of the elements in a work-group:
        //  {__local_tid * __data_per_work_item, __local_tid * __data_per_work_item + 1, ..., __wg_size * __data_per_work_item - 1}
        _LocOffsetSimdT __group_offset =
            __create_simd<_LocOffsetT, __data_per_work_item>(__local_tid * __data_per_work_item, 1);

        // The trick with IDs and the "fix" component in __global_fix is used to get relative indexes
        // of each digit in a work-group after reordering in SLM. Example when work-item id is 0:
        // key digit:              0  0  0  0  1  1  1  1  2  3
        // ----------------------------------------------------
        // key ID in a work-group: 0  1  2  3  4  5  6  7  8  9
        // "fix" component:        0  0  0  0 -4 -4 -4 -4 -8 -9
        // ----------------------------------------------------
        // digit offset:           0  1  2  3  0  1  2  3  0  0
        // Note: offset component from global histogram and the previous groups is also added as a part of "__global_fix" vector.
        _GlobOffsetSimdT __global_offset =
            __group_offset + __global_fix_lookup.template __lookup<__data_per_work_item>(__bins);

        __dpl_esimd::__vector_store<_KeyT, 1, __data_per_work_item>(
            __rng_data(__out_pack.__keys_rng()), __global_offset * sizeof(_KeyT), __pack.__keys, __global_offset < __n);
        if constexpr (__has_values)
        {
            __dpl_esimd::__vector_store<_ValT, 1, __data_per_work_item>(__rng_data(__out_pack.__vals_rng()),
                                                                        __global_offset * sizeof(_ValT), __pack.__vals,
                                                                        __global_offset < __n);
        }
    }

    [[sycl::reqd_sub_group_size(__sub_group_size)]] void
    operator()(sycl::nd_item<1> __idx) const
    {
        const ::std::uint32_t __local_tid = __idx.get_local_linear_id();
        const ::std::uint32_t __wg_id = __idx.get_group(0);
        const ::std::uint32_t __wg_size = __idx.get_local_range(0);
        const ::std::uint32_t __sg_id = __idx.get_sub_group().get_group_linear_id();
        const ::std::uint32_t __sg_local_id = __idx.get_sub_group().get_local_id();

        // TODO: Right now storing a full sub-group histogram contiguously in SLM. Consider evaluating
        // approach where each sub-group's bins are interleaved
        const ::std::uint32_t __sub_group_slm_offset = __sg_id * __bin_count;

        auto __values_simd_pack = __make_key_value_pack<__data_per_work_item, _KeyT, _ValT>();
        _LocOffsetT __bins[__data_per_work_item];
        _LocOffsetT __ranks[__data_per_work_item];
        _LocHistT __subgroup_offset;
        _GlobHistT __global_fix;

        __load_pack(__values_simd_pack, __wg_id, __sg_id, __sg_local_id);

        _ONEDPL_PRAGMA_UNROLL
        for (std::uint32_t __i = 0; __i < __data_per_work_item; ++__i)
        {
            const auto __ordered = __order_preserving_cast_scalar<__is_ascending>(__values_simd_pack.__keys[__i]);
            __bins[__i] = __get_bucket_scalar<__mask>(__ordered, __stage * __radix_bits);
        }
        // todo what's the best way to allocate SLM and unify with esimd impl?
        std::uint32_t* __slm;

        __rank_local(__idx, __ranks, __bins, __slm, __sub_group_slm_offset);
        //__rank_global(__subgroup_offset, __global_fix, __local_tid, __wg_id);

        //__reorder_reg_to_slm(__values_simd_pack, __ranks, __bins, __subgroup_offset, __wg_size, __sub_group_slm_offset);
        //__reorder_slm_to_glob(__values_simd_pack, __global_fix, __local_tid, __wg_size);
    }
};


} // namespace oneapi::dpl::experimental::kt::gpu::__impl

#endif // _ONEDPL_KT_SYCL_RADIX_SORT_KERNELS_H
