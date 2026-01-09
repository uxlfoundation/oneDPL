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

} // namespace oneapi::dpl::experimental::kt::gpu::__impl

#endif // _ONEDPL_KT_SYCL_RADIX_SORT_KERNELS_H
