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
    // Use uint32 for sub-group histograms (required for atomics), plus uint16 for scanning
    static constexpr std::uint32_t __bin_hist_slm_size_u32 = sizeof(std::uint32_t) * __bin_count * __num_sub_groups;
    static constexpr std::uint32_t __bin_hist_slm_size_u16 = sizeof(std::uint16_t) * __bin_count * __num_sub_groups;
    static constexpr std::uint32_t __incoming_offset_slm_size = (__bin_count + 1) * sizeof(std::uint16_t);
    // SLM layout: [uint32 histograms][uint16 histograms][incoming offsets] OR [reorder buffer]
    static constexpr std::uint32_t __slm_size =
        std::max(__reorder_slm_size, __bin_hist_slm_size_u32 + __bin_hist_slm_size_u16 + __incoming_offset_slm_size);

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

        const std::uint32_t __slm_reorder_this_thread = __local_tid * __data_per_work_item * sizeof(_KeyT);

        std::uint16_t* __slm = __slm_acc.get_multi_ptr<sycl::access::decorated::no>().get();

        // SLM layout: [uint32 sub-group histograms][uint16 converted histograms][incoming offsets]
        std::uint32_t* __slm_u32 = reinterpret_cast<std::uint32_t*>(__slm);
        std::uint32_t* __slm_sg_hist_u32 = &__slm_u32[__sub_group_id * __bin_count];
        _HistT* __slm_hist_u16 = reinterpret_cast<_HistT*>(__slm_u32 + __num_sub_groups * __bin_count);

        // Reduced register usage: only store local ranks, not full histograms
        _HistT __local_rank_in_bin[__data_per_work_item];
        _DeviceAddrT __write_addr[__data_per_work_item];
        _KeyT __keys[__data_per_work_item];
        _BinT __bins[__data_per_work_item];

        const _DeviceAddrT __io_offset = __data_per_work_item * __local_tid;
        // Optimization 1: Determine if this work-item has data (small-case optimization)
        const bool __has_data = (__io_offset < __n);

        // 1. Load data from the global memory to registers.
        auto __keys_in = __rng_data(__rng_pack_in.__keys_rng());
        _ONEDPL_PRAGMA_UNROLL
        for (std::uint32_t __i = 0; __i < __data_per_work_item; ++__i)
        {
            _DeviceAddrT __idx_global = __io_offset + __i;
            if (__idx_global < __n)
            {
                __keys[__i] = __keys_in[__idx_global];
            }
            else
            {
                __keys[__i] = __sort_identity<_KeyT, __is_ascending>();
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

            // 2.1 Initialize per-sub-group histogram in SLM (uint32 for atomics)
            // Optimization 2: Only zero bins that could have data (bounded zeroing)
            const std::uint32_t __max_bins_possible = std::min(__bin_count, __n);
            for (std::uint32_t __b = __sub_group_local_id; __b < __max_bins_possible; __b += __sub_group_size)
            {
                __slm_sg_hist_u32[__b] = 0;
            }
            __dpl_sycl::__group_barrier(__idx);

            // 2.2 Build histogram using atomics and capture local ranks
            // Optimization 1: Only active work-items perform atomic operations
            if (__has_data)
            {
                // Each work-item atomically increments bins and gets its rank within sub-group
                _ONEDPL_PRAGMA_UNROLL
                for (std::uint32_t __i = 0; __i < __data_per_work_item; ++__i)
                {
                    _BinT __bin = __bins[__i];
                    // Atomic fetch_add returns the OLD value, which is this work-item's rank
                    sycl::atomic_ref<std::uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group,
                                     sycl::access::address_space::local_space>
                        __atomic_ref(__slm_sg_hist_u32[__bin]);
                    __local_rank_in_bin[__i] = static_cast<_HistT>(__atomic_ref.fetch_add(1));
                }
            }
            __dpl_sycl::__group_barrier(__idx);

            // 2.3 Convert uint32 histograms to uint16 for scanning
            // Cooperatively copy to separate uint16 area
            // Note: inactive sub-groups will have all-zero histograms from the atomics phase
            for (std::uint32_t __b = __sub_group_local_id; __b < __bin_count; __b += __sub_group_size)
            {
                __slm_hist_u16[__sub_group_id * __bin_count + __b] = static_cast<_HistT>(__slm_sg_hist_u32[__b]);
            }
            __dpl_sycl::__group_barrier(__idx);

            // 2.4. Vector scan of uint16 histograms previously accumulated by each sub-group.
            // Use the first __bin_summary_sub_group_size sub-groups to do the scan work cooperatively
            constexpr std::uint32_t __bin_summary_sub_group_size = __bin_count / __sub_group_size;
            static_assert(__bin_count % __sub_group_size == 0);

            if (__sub_group_id < __bin_summary_sub_group_size)
            {
                constexpr std::uint32_t __bin_width = __sub_group_size;
                std::uint32_t __slm_bin_hist_summary_offset = __sub_group_id * __bin_width + __sub_group_local_id;
                _HistT __item_grf_hist_summary = 0;

                // 2.4.1 Vector scan of the same bins across different sub-group histograms.
                __item_grf_hist_summary = __slm_hist_u16[__slm_bin_hist_summary_offset];
                __slm_bin_hist_summary_offset += __bin_count;

                for (std::uint32_t __s = 1; __s < __num_sub_groups - 1; __s++)
                {
                    _HistT __tmp = __slm_hist_u16[__slm_bin_hist_summary_offset];
                    __item_grf_hist_summary += __tmp;
                    __slm_hist_u16[__slm_bin_hist_summary_offset] = __item_grf_hist_summary;
                    __slm_bin_hist_summary_offset += __bin_count;
                }

                // 2.4.2 Vector scan of different bins inside one histogram, the final one for the whole work-group.
                _HistT __tmp = __slm_hist_u16[__slm_bin_hist_summary_offset];
                __item_grf_hist_summary += __tmp;

                // Perform sub-group scan on the summary - each work-item has one bin's count
                _HistT __item_grf_hist_summary_arr[1] = {__item_grf_hist_summary};
                __sub_group_scan<__sub_group_size, 1>(__idx.get_sub_group(), __item_grf_hist_summary_arr, std::plus<>{},
                                                      __bin_width);

                __slm_hist_u16[__slm_bin_hist_summary_offset] = __item_grf_hist_summary_arr[0];
            }
            __dpl_sycl::__group_barrier(__idx);

            // 2.4.3 One sub-group finalizes scan performed at stage 2.4.2
            // by propagating prefixes accumulated after scanning individual "__bin_width" pieces.
            if (__sub_group_id == __bin_summary_sub_group_size)
            {
                // Use the final sub-group histogram position as the data to finalize
                _HistT* __scan_elements = &__slm_hist_u16[(__num_sub_groups - 1) * __bin_count];

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
                    __slm_hist_u16[__slm_incoming_offset + __i * __sub_group_size + __sub_group_local_id] =
                        __scan_elements[__i * __sub_group_size + __sub_group_local_id];
                }
            }
            __dpl_sycl::__group_barrier(__idx);

            // 2.5. Compute total offsets using atomic-assigned local ranks
            // Load scanned global offsets and combine with local ranks and previous sub-group offsets
            // Optimization 1: Only active work-items compute offsets
            if (__has_data)
            {
                std::uint32_t __slm_incoming_offset = __bin_hist_slm_size_u16 / sizeof(_HistT);

                _ONEDPL_PRAGMA_UNROLL
                for (std::uint32_t __i = 0; __i < __data_per_work_item; ++__i)
                {
                    _BinT __bin = __bins[__i];
                    // Get global + previous sub-group offset
                    _HistT __global_and_prev_sg_offset = __slm_hist_u16[__slm_incoming_offset + __bin];
                    if (__sub_group_id > 0)
                    {
                        __global_and_prev_sg_offset += __slm_hist_u16[(__sub_group_id - 1) * __bin_count + __bin];
                    }
                    // Combine: local_rank (from atomic) + previous_sub_groups + global
                    __write_addr[__i] = __local_rank_in_bin[__i] + __global_and_prev_sg_offset;
                }
            }

            // 2.7. Reorder keys in SLM.
            if (__stage != __stage_count - 1)
            {
                _KeyT* __slm_keys = reinterpret_cast<_KeyT*>(__slm);
                // Optimization 1: Only active work-items write to SLM
                if (__has_data)
                {
                    _ONEDPL_PRAGMA_UNROLL
                    for (std::uint32_t __i = 0; __i < __data_per_work_item; ++__i)
                    {
                        __slm_keys[__write_addr[__i]] = __keys[__i];
                    }
                }
                __dpl_sycl::__group_barrier(__idx);

                // Read keys back from SLM in sorted order
                // Optimization 1: Only active work-items read from SLM
                if (__has_data)
                {
                    _ONEDPL_PRAGMA_UNROLL
                    for (std::uint32_t __i = 0; __i < __data_per_work_item; ++__i)
                    {
                        __keys[__i] = __slm_keys[__slm_reorder_this_thread / sizeof(_KeyT) + __i];
                    }
                }
            }
        }

        // 3. Store keys to the global memory.
        // Optimization 1: Only active work-items write to global memory
        if (__has_data)
        {
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
    }
};

} // namespace oneapi::dpl::experimental::kt::gpu::__impl

#endif // _ONEDPL_KT_SYCL_RADIX_SORT_ONE_WG_KERNEL_H
