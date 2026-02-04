// -*- C++ -*-
//===-- sycl_radix_sort_one_wg_kernel.h ----------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _ONEDPL_KT_SYCL_RADIX_SORT_ONE_WG_KERNEL_H
#define _ONEDPL_KT_SYCL_RADIX_SORT_ONE_WG_KERNEL_H

#include <cstdint>
#include <type_traits>

#include "../../../pstl/hetero/dpcpp/sycl_defs.h"
#include "../../../pstl/utils.h"
#include "../../../pstl/hetero/dpcpp/utils_ranges_sycl.h"

#include "sub_group/sub_group_scan.h"
#include "radix_sort_utils.h"

namespace oneapi::dpl::experimental::kt::gpu::__impl
{

//-----------------------------------------------------------------------------
// SYCL one-work-group kernel struct
//-----------------------------------------------------------------------------
template <bool __is_ascending, std::uint8_t __radix_bits, std::uint16_t __data_per_work_item,
          std::uint16_t __work_group_size, typename _KeyT, typename _RngPack1, typename _RngPack2>
struct __one_wg_kernel<__sycl_tag, __is_ascending, __radix_bits, __data_per_work_item, __work_group_size, _KeyT,
                       _RngPack1, _RngPack2>
{
    static constexpr std::uint32_t __sub_group_size = 32;
    static constexpr std::uint32_t __bin_count = 1 << __radix_bits;
    static constexpr std::uint32_t __num_sub_groups = __work_group_size / __sub_group_size;
    static constexpr std::uint32_t __reorder_slm_size = __data_per_work_item * sizeof(_KeyT) * __work_group_size;
    // Use uint16 for sub-group histograms (ballot-based ranking, no atomics needed)
    static constexpr std::uint32_t __bin_hist_slm_size_u16 = sizeof(std::uint16_t) * __bin_count * __num_sub_groups;
    static constexpr std::uint32_t __incoming_offset_slm_size = (__bin_count + 1) * sizeof(std::uint16_t);
    // SLM layout: [uint16 histograms][incoming offsets] OR [reorder buffer]
    static constexpr std::uint32_t __slm_size =
        std::max(__reorder_slm_size, __bin_hist_slm_size_u16 + __incoming_offset_slm_size);

    std::uint32_t __n;
    _RngPack1 __rng_pack_in;
    _RngPack2 __rng_pack_out;
    sycl::local_accessor<std::uint16_t, 1> __slm_acc;

    __one_wg_kernel(std::uint32_t __n, const _RngPack1& __rng_pack_in, const _RngPack2& __rng_pack_out,
                    sycl::local_accessor<std::uint16_t, 1> __slm_accessor)
        : __n(__n), __rng_pack_in(__rng_pack_in), __rng_pack_out(__rng_pack_out), __slm_acc(__slm_accessor)
    {
    }

    [[sycl::reqd_sub_group_size(__sub_group_size)]] void
    operator()(sycl::nd_item<1> __idx) const
    {
        using _BinT = std::uint16_t;
        using _HistT = std::uint16_t;
        using _DeviceAddrT = std::uint32_t;
        using _LocIdxT = std::uint16_t;

        constexpr std::uint32_t __bit_count = sizeof(_KeyT) * 8;
        constexpr std::uint32_t __stage_count = oneapi::dpl::__internal::__dpl_ceiling_div(__bit_count, __radix_bits);
        constexpr _BinT __mask = __bin_count - 1;
        constexpr std::uint32_t __hist_stride = sizeof(_HistT) * __bin_count;

        const std::uint32_t __local_tid = __idx.get_local_linear_id();
        const std::uint32_t __sub_group_id = __idx.get_sub_group().get_group_linear_id();
        const std::uint32_t __sub_group_local_id = __idx.get_sub_group().get_local_linear_id();

        std::uint16_t* __slm = __slm_acc.get_multi_ptr<sycl::access::decorated::no>().get();

        // SLM layout: [uint16 sub-group histograms][incoming offsets]
        _HistT* __slm_sg_hist_u16 = &__slm[__sub_group_id * __bin_count];

        // Reduced register usage: only store local ranks, not full histograms
        _HistT __local_rank_in_bin[__data_per_work_item];
        _DeviceAddrT __write_addr[__data_per_work_item];
        _KeyT __keys[__data_per_work_item];
        _BinT __bins[__data_per_work_item];

        // 1. Load data from the global memory to registers with sub-group stride
        const _DeviceAddrT __sub_group_start = __sub_group_id * __data_per_work_item * __sub_group_size;
        const _DeviceAddrT __wi_offset = __sub_group_start + __sub_group_local_id;
        const bool __is_full_block = (__sub_group_start + __data_per_work_item * __sub_group_size) <= __n;
        
        auto __keys_in = __rng_data(__rng_pack_in.__keys_rng());
        if (__is_full_block)
        {
            _ONEDPL_PRAGMA_UNROLL
            for (std::uint32_t __i = 0; __i < __data_per_work_item; ++__i)
            {
                __keys[__i] = __keys_in[__i * __sub_group_size + __wi_offset];
            }
        }
        else
        {
            for (std::uint32_t __i = 0; __i < __data_per_work_item; ++__i)
            {
                _DeviceAddrT __idx_global = __i * __sub_group_size + __wi_offset;
                if (__idx_global < __n)
                {
                    __keys[__i] = __keys_in[__idx_global];
                }
                else
                {
                    __keys[__i] = __sort_identity<_KeyT, __is_ascending>();
                }
            }
        }

        // 2. Sort each __radix_bits
        for (std::uint32_t __stage = 0; __stage < __stage_count; __stage++)
        {
            // Compute bins for this stage
            _ONEDPL_PRAGMA_UNROLL
            for (std::uint32_t __i = 0; __i < __data_per_work_item; ++__i)
            {
                __bins[__i] = __get_bucket_scalar<__mask>(__order_preserving_cast_scalar<__is_ascending>(__keys[__i]),
                                                          __stage * __radix_bits);
            }

            // 2.1 Initialize all sub-group histograms in SLM
            for (std::uint32_t __b = __local_tid; __b < __num_sub_groups * __bin_count; __b += __work_group_size)
            {
                __slm[__b] = 0;
            }
            __dpl_sycl::__group_barrier(__idx);

            // 2.2 Build histogram using ballot-based ranking (similar to __rank_local)
            constexpr std::uint32_t __sub_group_full_bitmask = 0x7fffffff;
            static_assert(__sub_group_size == 32);
            // lower bits than my current will be set meaning we only preserve left lanes
            std::uint32_t __remove_right_lanes = __sub_group_full_bitmask >> (__sub_group_size - 1 - __sub_group_local_id);

            _ONEDPL_PRAGMA_UNROLL
            for (std::uint32_t __i = 0; __i < __data_per_work_item; ++__i)
            {
                _BinT __bin = __bins[__i];
                
                // Match bins using ballot
                sycl::ext::oneapi::sub_group_mask __matched_bins = sycl::ext::oneapi::group_ballot(__idx.get_sub_group());
                _ONEDPL_PRAGMA_UNROLL
                for (int __b = 0; __b < __radix_bits; __b++)
                {
                    bool __bit = static_cast<bool>((__bin >> __b) & 1);
                    sycl::ext::oneapi::sub_group_mask __sg_vote = sycl::ext::oneapi::group_ballot(__idx.get_sub_group(), __bit);
                    __matched_bins &= __bit ? __sg_vote : ~__sg_vote;
                }
                std::uint32_t __matched_bins_mask = 0;
                __matched_bins.extract_bits(__matched_bins_mask);
                
                _HistT __pre_rank = __slm_sg_hist_u16[__bin];
                std::uint32_t __matched_left_lanes = __matched_bins_mask & __remove_right_lanes;
                _HistT __this_round_rank = sycl::popcount(__matched_left_lanes);
                _HistT __this_round_count = sycl::popcount(__matched_bins_mask);
                _HistT __rank_after = __pre_rank + __this_round_rank;
                bool __is_leader = __this_round_rank == __this_round_count - 1;
                if (__is_leader)
                {
                    __slm_sg_hist_u16[__bin] = __rank_after + 1;
                }
                __local_rank_in_bin[__i] = __rank_after;
            }
            __dpl_sycl::__group_barrier(__idx);

            // 2.3 Vector scan of uint16 histograms previously accumulated by each sub-group.
            // Use the first __bin_summary_sub_group_size sub-groups to do the scan work cooperatively
            constexpr std::uint32_t __bin_summary_sub_group_size = __bin_count / __sub_group_size;
            static_assert(__bin_count % __sub_group_size == 0);

            if (__sub_group_id < __bin_summary_sub_group_size)
            {
                constexpr std::uint32_t __bin_width = __sub_group_size;
                _HistT __item_grf_hist_summary = 0;

                // 2.4.1 Vector scan of the same bins across different sub-group histograms.
                std::uint32_t __slm_bin_hist_summary_offset = __sub_group_id * __bin_width;

                for (std::uint32_t __s = 0; __s < __num_sub_groups; __s++, __slm_bin_hist_summary_offset += __bin_count)
                {
                    std::uint32_t __slm_idx = __slm_bin_hist_summary_offset + __sub_group_local_id;
                    __item_grf_hist_summary += __slm[__slm_idx];
                    __slm[__slm_idx] = __item_grf_hist_summary;
                }

                // 2.4.2 Vector scan of different bins inside one histogram, the final one for the whole work-group.
                // Perform sub-group scan on the summary - each work-item has one bin's count
                _HistT __item_grf_hist_summary_arr[1] = {__item_grf_hist_summary};
                __sub_group_scan<__sub_group_size, 1>(__idx.get_sub_group(), __item_grf_hist_summary_arr, std::plus<>{},
                                                      __bin_width);

                __slm[__slm_bin_hist_summary_offset] = __item_grf_hist_summary_arr[0];
            }
            __dpl_sycl::__group_barrier(__idx);

            // 2.4.3 One sub-group finalizes scan performed at stage 2.4.2
            // by propagating prefixes accumulated after scanning individual "__bin_width" pieces.
            if (__sub_group_id == __bin_summary_sub_group_size)
            {
                // Use the final sub-group histogram position as the data to finalize
                _HistT* __scan_elements = &__slm[(__num_sub_groups - 1) * __bin_count];

                // Cross-segment exclusive scan
                _HistT __carry = 0;
                _ONEDPL_PRAGMA_UNROLL
                for (std::uint32_t __i = 0; __i < __bin_summary_sub_group_size; ++__i)
                {
                    auto __element = __scan_elements[__i * __sub_group_size + __sub_group_local_id];
                    auto __element_right_shift = sycl::shift_group_right(__idx.get_sub_group(), __element, 1);
                    if (__sub_group_local_id == 0)
                        __element_right_shift = 0;
                    __scan_elements[__i * __sub_group_size + __sub_group_local_id] = __element_right_shift + __carry;

                    __carry += sycl::group_broadcast(__idx.get_sub_group(), __element, __sub_group_size - 1);
                }

                // Store back to SLM at the incoming offset location (after uint16 histograms)
                // IMPORTANT: Store ALL bins (all __bin_summary_sub_group_size segments)
                std::uint32_t __slm_incoming_offset = __bin_hist_slm_size_u16 / sizeof(_HistT);
                _ONEDPL_PRAGMA_UNROLL
                for (std::uint32_t __i = 0; __i < __bin_summary_sub_group_size; ++__i)
                {
                    __slm[__slm_incoming_offset + __i * __sub_group_size + __sub_group_local_id] =
                        __scan_elements[__i * __sub_group_size + __sub_group_local_id];
                }
            }
            __dpl_sycl::__group_barrier(__idx);

            // 2.5. Compute total offsets using ballot-assigned local ranks
            // The scanned histogram contains exclusive prefix sums (starting positions for each bin)
            std::uint32_t __slm_incoming_offset = __bin_hist_slm_size_u16 / sizeof(_HistT);

            _ONEDPL_PRAGMA_UNROLL
            for (std::uint32_t __i = 0; __i < __data_per_work_item; ++__i)
            {
                _BinT __bin = __bins[__i];
                // Get exclusive prefix sum for this bin (start position in reordered array)
                _HistT __bin_start = __slm[__slm_incoming_offset + __bin];
                // Add the contribution of elements in this bin from previous sub-groups
                _HistT __prev_sg_count_for_bin = 0;
                for (std::uint32_t __sg = 0; __sg < __sub_group_id; ++__sg)
                {
                    __prev_sg_count_for_bin += __slm[__sg * __bin_count + __bin];
                }
                // Final address: bin_start + previous_subgroups_count_for_this_bin + local_rank
                __write_addr[__i] = __bin_start + __prev_sg_count_for_bin + __local_rank_in_bin[__i];
            }

            // 2.5. Reorder keys in SLM.
            if (__stage != __stage_count - 1)
            {
                _KeyT* __slm_keys = reinterpret_cast<_KeyT*>(__slm);
                
                _ONEDPL_PRAGMA_UNROLL
                for (std::uint32_t __i = 0; __i < __data_per_work_item; ++__i)
                {
                    __slm_keys[__write_addr[__i]] = __keys[__i];
                }
                __dpl_sycl::__group_barrier(__idx);

                // Read keys back from SLM in sub-group strided order
                const _DeviceAddrT __keys_slm_offset = __sub_group_id * __data_per_work_item * __sub_group_size;
                _ONEDPL_PRAGMA_UNROLL
                for (std::uint32_t __i = 0; __i < __data_per_work_item; ++__i)
                {
                    _LocIdxT __slm_idx = __keys_slm_offset + __i * __sub_group_size + __sub_group_local_id;
                    __keys[__i] = __slm_keys[__slm_idx];
                }
                __dpl_sycl::__group_barrier(__idx);
            }
        }

        // 3. Store keys to the global memory.
        auto __keys_out = __rng_data(__rng_pack_out.__keys_rng());
        _ONEDPL_PRAGMA_UNROLL
        for (std::uint32_t __i = 0; __i < __data_per_work_item; ++__i)
        {
            if (__write_addr[__i] < __n)
            {
                __keys_out[__write_addr[__i]] = __keys[__i];
            }
        }
    }
};

} // namespace oneapi::dpl::experimental::kt::gpu::__impl

#endif // _ONEDPL_KT_SYCL_RADIX_SORT_ONE_WG_KERNEL_H
