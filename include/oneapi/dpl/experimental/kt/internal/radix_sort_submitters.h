// -*- C++ -*-
//===-- radix_sort_submitters.h --------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===------------------------------------------------------===//

#ifndef _ONEDPL_KT_RADIX_SORT_SUBMITTERS_H
#define _ONEDPL_KT_RADIX_SORT_SUBMITTERS_H

#include <cstdint>
#include <utility>
#include <type_traits>

#include "../../../pstl/hetero/dpcpp/sycl_defs.h"

#include "../../../pstl/hetero/dpcpp/utils_ranges_sycl.h"
#include "../../../pstl/hetero/dpcpp/parallel_backend_sycl_utils.h"
#include "../../../pstl/hetero/dpcpp/sycl_traits.h" //SYCL traits specialization for some oneDPL types.

#include "esimd_radix_sort_kernels.h"
#include "sycl_radix_sort_kernels.h"
#include "esimd_defs.h"
#include "../../../pstl/hetero/dpcpp/parallel_backend_sycl_radix_sort_one_wg.h"

namespace oneapi::dpl::experimental::kt::gpu::__impl
{

//------------------------------------------------------------------------
// Please see the comment above __parallel_for_small_submitter for optional kernel name explanation
//------------------------------------------------------------------------

// Kernel name tag for SYCL one work group sort
template <typename... _Name>
struct __sycl_radix_sort_one_wg_kernel_name;

template <bool __is_ascending, ::std::uint8_t __radix_bits, ::std::uint16_t __data_per_work_item,
          ::std::uint16_t __work_group_size, typename _KeyT, typename _KernelName>
struct __radix_sort_one_wg_submitter;

template <bool __is_ascending, ::std::uint8_t __radix_bits, ::std::uint16_t __data_per_work_item,
          ::std::uint16_t __work_group_size, typename _KeyT, typename... _Name>
struct __radix_sort_one_wg_submitter<__is_ascending, __radix_bits, __data_per_work_item, __work_group_size, _KeyT,
                                     oneapi::dpl::__par_backend_hetero::__internal::__optional_kernel_name<_Name...>>
{
    template <typename _RngPack1, typename _RngPack2>
    sycl::event
    operator()(__esimd_tag, sycl::queue __q, _RngPack1&& __pack_in, _RngPack2&& __pack_out, ::std::size_t __n) const
    {
        sycl::nd_range<1> __nd_range{__work_group_size, __work_group_size};
        return __q.submit([&](sycl::handler& __cgh) {
            oneapi::dpl::__ranges::__require_access(__cgh, __pack_in.__keys_rng());
            oneapi::dpl::__ranges::__require_access(__cgh, __pack_out.__keys_rng());
            __one_wg_kernel<__esimd_tag, __is_ascending, __radix_bits, __data_per_work_item, __work_group_size, _KeyT,
                            std::decay_t<_RngPack1>, std::decay_t<_RngPack2>>
                __kernel(__n, __pack_in, __pack_out);
            __cgh.parallel_for<_Name...>(__nd_range, __kernel);
        });
    }

    template <typename _RngPack1, typename _RngPack2>
    sycl::event
    operator()(__sycl_tag, sycl::queue __q, _RngPack1&& __pack_in, _RngPack2&& __pack_out, ::std::size_t __n) const
    {
        // Use __subgroup_radix_sort with default radix=4 and block_size=__data_per_work_item
        constexpr ::std::uint16_t __block_size = __data_per_work_item;
        constexpr ::std::uint32_t __radix = 4;

        // Create a unique kernel name using __kernel_name_provider
        // Include range pack types to ensure uniqueness across different invocations
        using _KernelName = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<
            __sycl_radix_sort_one_wg_kernel_name<_Name..., std::decay_t<_RngPack1>, std::decay_t<_RngPack2>>>;

        using _SubgroupRadixSort =
            oneapi::dpl::__par_backend_hetero::__subgroup_radix_sort<_KernelName, __work_group_size, __block_size,
                                                                     __radix, __is_ascending>;

        _SubgroupRadixSort __sorter;

        // Now sort the output range in-place using identity projection
        auto __identity_proj = [](const _KeyT& __x) { return __x; };
        return __sorter(__q, __pack_in.__keys_rng(), __pack_out.__keys_rng(), __identity_proj);
    }
};

template <bool __is_ascending, ::std::uint8_t __radix_bits, ::std::uint32_t __hist_work_group_count,
          ::std::uint16_t __hist_work_group_size, typename _KernelName>
struct __radix_sort_histogram_submitter;

template <bool __is_ascending, ::std::uint8_t __radix_bits, ::std::uint32_t __hist_work_group_count,
          ::std::uint16_t __hist_work_group_size, typename... _Name>
struct __radix_sort_histogram_submitter<__is_ascending, __radix_bits, __hist_work_group_count, __hist_work_group_size,
                                        oneapi::dpl::__par_backend_hetero::__internal::__optional_kernel_name<_Name...>>
{
    template <typename _KeysRng, typename _GlobalOffsetData>
    sycl::event
    operator()(__esimd_tag, sycl::queue& __q, const _KeysRng& __keys_rng, const _GlobalOffsetData& __global_offset_data,
               ::std::size_t __n, const sycl::event& __e) const
    {
        sycl::nd_range<1> __nd_range(__hist_work_group_count * __hist_work_group_size, __hist_work_group_size);
        return __q.submit([&](sycl::handler& __cgh) {
            oneapi::dpl::__ranges::__require_access(__cgh, __keys_rng);
            __cgh.depends_on(__e);
            __global_histogram<__esimd_tag, __is_ascending, __radix_bits, __hist_work_group_count,
                               __hist_work_group_size, std::decay_t<_KeysRng>>
                __kernel(__n, __keys_rng, __global_offset_data);
            __cgh.parallel_for<_Name...>(__nd_range, __kernel);
        });
    }

    template <typename _KeysRng, typename _GlobalOffsetData>
    sycl::event
    operator()(__sycl_tag, sycl::queue& __q, const _KeysRng& __keys_rng, const _GlobalOffsetData& __global_offset_data,
               ::std::size_t __n, const sycl::event& __e) const
    {
        using _KeyT = oneapi::dpl::__internal::__value_t<_KeysRng>;
        constexpr std::uint32_t __sub_group_size = 32;
        constexpr std::uint32_t __bit_count = sizeof(_KeyT) * 8;
        constexpr std::uint32_t __bin_count = 1 << __radix_bits;
        constexpr std::uint32_t __stage_count = oneapi::dpl::__internal::__dpl_ceiling_div(__bit_count, __radix_bits);
        constexpr std::uint32_t __hist_buffer_size = __stage_count * __bin_count;

        // Calculate number of histograms based on SLM capacity
        constexpr std::uint32_t __max_histograms = 16;
        constexpr std::uint32_t __max_slm_bytes = 1 << 16;
        constexpr std::uint32_t __num_histograms =
            std::min(__max_histograms, std::uint32_t(__max_slm_bytes / (__hist_buffer_size * sizeof(std::uint32_t))));

        sycl::nd_range<1> __nd_range(__hist_work_group_count * __hist_work_group_size, __hist_work_group_size);
        return __q.submit([&](sycl::handler& __cgh) {
            sycl::local_accessor<std::uint32_t, 1> __slm_accessor(__hist_buffer_size * __num_histograms, __cgh);
            oneapi::dpl::__ranges::__require_access(__cgh, __keys_rng);
            __cgh.depends_on(__e);
            __global_histogram<__sycl_tag, __is_ascending, __radix_bits, __hist_work_group_count,
                               __hist_work_group_size, std::decay_t<_KeysRng>>
                __kernel(__n, __keys_rng, __slm_accessor, __global_offset_data, __num_histograms);
            __cgh.parallel_for<_Name...>(__nd_range, __kernel);
        });
    }
};

template <::std::uint32_t __stage_count, ::std::uint16_t __bin_count, typename _KernelName>
struct __radix_sort_onesweep_scan_submitter;

template <::std::uint32_t __stage_count, ::std::uint32_t __bin_count, typename... _Name>
struct __radix_sort_onesweep_scan_submitter<
    __stage_count, __bin_count, oneapi::dpl::__par_backend_hetero::__internal::__optional_kernel_name<_Name...>>
{
    template <typename _KtTag, typename _GlobalOffsetData>
    sycl::event
    operator()(_KtTag, sycl::queue& __q, const _GlobalOffsetData& __global_offset_data, const sycl::event& __e) const
    {
        //scan kernel is pure sycl for esimd_sort, so no need to dispatch from tag
        sycl::nd_range<1> __nd_range(__stage_count * __bin_count, __bin_count);
        return __q.submit([&](sycl::handler& __cgh) {
            __cgh.depends_on(__e);
            __cgh.parallel_for<_Name...>(__nd_range, [=](sycl::nd_item<1> __nd_item) {
                ::std::uint32_t __offset = __nd_item.get_global_id(0);
                const auto __g = __nd_item.get_group();
                ::std::uint32_t __count = __global_offset_data[__offset];
                ::std::uint32_t __presum =
                    __dpl_sycl::__exclusive_scan_over_group(__g, __count, __dpl_sycl::__plus<::std::uint32_t>());
                __global_offset_data[__offset] = __presum;
            });
        });
    }
};

template <bool __is_ascending, ::std::uint8_t __radix_bits, ::std::uint16_t __data_per_work_item,
          ::std::uint16_t __work_group_size, typename _KernelName>
struct __radix_sort_onesweep_submitter;

template <bool __is_ascending, ::std::uint8_t __radix_bits, ::std::uint16_t __data_per_work_item,
          ::std::uint16_t __work_group_size, typename... _Name>
struct __radix_sort_onesweep_submitter<__is_ascending, __radix_bits, __data_per_work_item, __work_group_size,
                                       oneapi::dpl::__par_backend_hetero::__internal::__optional_kernel_name<_Name...>>
{
    template <typename _InRngPack, typename _OutRngPack, typename _GlobalHistT, typename _AtomicIdT>
    sycl::event
    operator()(__esimd_tag, sycl::queue& __q, _InRngPack&& __in_pack, _OutRngPack&& __out_pack,
               _GlobalHistT* __p_global_hist, _GlobalHistT* __p_group_hists, _AtomicIdT* __p_atomic_id,
               ::std::uint32_t __sweep_work_group_count, ::std::size_t __n, ::std::uint32_t __stage,
               const sycl::event& __e) const
    {
        sycl::nd_range<1> __nd_range(__sweep_work_group_count * __work_group_size, __work_group_size);
        return __q.submit([&](sycl::handler& __cgh) {
            oneapi::dpl::__ranges::__require_access(__cgh, __in_pack.__keys_rng(), __out_pack.__keys_rng());
            if constexpr (::std::decay_t<_InRngPack>::__has_values)
            {
                oneapi::dpl::__ranges::__require_access(__cgh, __in_pack.__vals_rng(), __out_pack.__vals_rng());
            }
            __cgh.depends_on(__e);
            __radix_sort_onesweep_kernel<__esimd_tag, __is_ascending, __radix_bits, __data_per_work_item,
                                         __work_group_size, ::std::decay_t<_InRngPack>, ::std::decay_t<_OutRngPack>>
                __kernel(__n, __stage, __p_global_hist, __p_group_hists, __p_atomic_id,
                         ::std::forward<_InRngPack>(__in_pack), ::std::forward<_OutRngPack>(__out_pack));
            __cgh.parallel_for<_Name...>(__nd_range, __kernel);
        });
    }

    template <typename _InRngPack, typename _OutRngPack, typename _GlobalHistT, typename _AtomicIdT>
    sycl::event
    operator()(__sycl_tag, sycl::queue& __q, _InRngPack&& __in_pack, _OutRngPack&& __out_pack,
               _GlobalHistT* __p_global_hist, _GlobalHistT* __p_group_hists, _AtomicIdT* __p_atomic_id,
               ::std::uint32_t __sweep_work_group_count, ::std::size_t __n, ::std::uint32_t __stage,
               const sycl::event& __e) const
    {
        using _KernelType =
            __radix_sort_onesweep_kernel<__sycl_tag, __is_ascending, __radix_bits, __data_per_work_item,
                                         __work_group_size, ::std::decay_t<_InRngPack>, ::std::decay_t<_OutRngPack>>;
        constexpr ::std::uint32_t __slm_size_bytes = _KernelType::__calc_slm_alloc();
        constexpr ::std::uint32_t __slm_size_elements = __slm_size_bytes / sizeof(::std::uint32_t);

        sycl::nd_range<1> __nd_range(__sweep_work_group_count * __work_group_size, __work_group_size);
        return __q.submit([&](sycl::handler& __cgh) {
            sycl::local_accessor<unsigned char, 1> __slm_accessor(__slm_size_bytes, __cgh);
            oneapi::dpl::__ranges::__require_access(__cgh, __in_pack.__keys_rng(), __out_pack.__keys_rng());
            if constexpr (::std::decay_t<_InRngPack>::__has_values)
            {
                oneapi::dpl::__ranges::__require_access(__cgh, __in_pack.__vals_rng(), __out_pack.__vals_rng());
            }
            __cgh.depends_on(__e);
            _KernelType __kernel(__n, __stage, __p_global_hist, __p_group_hists, __p_atomic_id,
                                 ::std::forward<_InRngPack>(__in_pack), ::std::forward<_OutRngPack>(__out_pack),
                                 __slm_accessor);
            __cgh.parallel_for<_Name...>(__nd_range, __kernel);
        });
    }
};

template <typename _KernelName>
struct __radix_sort_copyback_submitter;

template <typename... _Name>
struct __radix_sort_copyback_submitter<oneapi::dpl::__par_backend_hetero::__internal::__optional_kernel_name<_Name...>>
{
    template <typename _KtTag, typename _InRngPack, typename _OutRngPack>
    sycl::event
    operator()(_KtTag, sycl::queue& __q, _InRngPack&& __in_pack, _OutRngPack&& __out_pack, ::std::uint32_t __n,
               const sycl::event& __e) const
    {
        //copyback kernel is pure sycl for esimd_sort, so no need to dispatch from tag
        return __q.submit([&](sycl::handler& __cgh) {
            oneapi::dpl::__ranges::__require_access(__cgh, __in_pack.__keys_rng(), __out_pack.__keys_rng());
            if constexpr (::std::decay_t<_InRngPack>::__has_values)
            {
                oneapi::dpl::__ranges::__require_access(__cgh, __in_pack.__vals_rng(), __out_pack.__vals_rng());
            }
            // TODO: make sure that access is read_only for __keys_tmp_rng/__vals_tmp_rng  and is write_only for __keys_rng/__vals_rng
            __cgh.depends_on(__e);
            __cgh.parallel_for<_Name...>(sycl::range<1>{__n}, [=](sycl::item<1> __item) {
                auto __global_id = __item.get_linear_id();
                __rng_data(__out_pack.__keys_rng())[__global_id] = __rng_data(__in_pack.__keys_rng())[__global_id];
                if constexpr (::std::decay_t<_InRngPack>::__has_values)
                {
                    __rng_data(__out_pack.__vals_rng())[__global_id] = __rng_data(__in_pack.__vals_rng())[__global_id];
                }
            });
        });
    }
};

} // namespace oneapi::dpl::experimental::kt::gpu::__impl

#endif // _ONEDPL_KT_RADIX_SORT_SUBMITTERS_H
