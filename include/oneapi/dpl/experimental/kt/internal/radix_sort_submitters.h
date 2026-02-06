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

template <bool __is_ascending, std::uint8_t __radix_bits, std::uint16_t __data_per_work_item,
          std::uint16_t __work_group_size, typename _KeyT, typename _KernelName>
struct __radix_sort_one_wg_submitter;

template <bool __is_ascending, std::uint8_t __radix_bits, std::uint16_t __data_per_work_item,
          std::uint16_t __work_group_size, typename _KeyT, typename... _Name>
struct __radix_sort_one_wg_submitter<__is_ascending, __radix_bits, __data_per_work_item, __work_group_size, _KeyT,
                                     oneapi::dpl::__par_backend_hetero::__internal::__optional_kernel_name<_Name...>>
{
    template <typename _RngPack1, typename _RngPack2>
    sycl::event
    operator()(__esimd_tag, sycl::queue __q, _RngPack1&& __pack_in, _RngPack2&& __pack_out, std::size_t __n) const
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
    operator()(__sycl_tag, sycl::queue __q, _RngPack1&& __pack_in, _RngPack2&& __pack_out, std::size_t __n) const
    {
        // TODO: Use user-provided work-group sizes and data per work item. However, 8-bit radix is broken in oneDPL, so we
        // must force 4-bit for now.
        constexpr std::uint16_t __block_size = __data_per_work_item;
        constexpr std::uint32_t __radix = 4;

        // Create a unique kernel name using __kernel_name_provider
        // Include range pack types to ensure uniqueness across different invocations
        using _KernelName = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<
            __sycl_radix_sort_one_wg_kernel_name<_Name..., std::decay_t<_RngPack1>, std::decay_t<_RngPack2>>>;

        using _SubgroupRadixSort =
            oneapi::dpl::__par_backend_hetero::__subgroup_radix_sort<_KernelName, __work_group_size, __block_size,
                                                                     __radix, __is_ascending>;

        _SubgroupRadixSort __sorter;

        auto __identity_proj = [](const _KeyT& __x) { return __x; };
        return __sorter(__q, __pack_in.__keys_rng(), __pack_out.__keys_rng(), __identity_proj);
    }
};

template <bool __is_ascending, std::uint8_t __radix_bits, std::uint32_t __hist_work_group_count,
          std::uint16_t __hist_work_group_size, typename _KernelName>
struct __radix_sort_histogram_submitter;

template <bool __is_ascending, std::uint8_t __radix_bits, std::uint32_t __hist_work_group_count,
          std::uint16_t __hist_work_group_size, typename... _Name>
struct __radix_sort_histogram_submitter<__is_ascending, __radix_bits, __hist_work_group_count, __hist_work_group_size,
                                        oneapi::dpl::__par_backend_hetero::__internal::__optional_kernel_name<_Name...>>
{
    template <typename _KeysRng, typename _GlobalOffsetData>
    sycl::event
    operator()(__esimd_tag, sycl::queue& __q, const _KeysRng& __keys_rng, const _GlobalOffsetData& __global_offset_data,
               std::size_t __n, const sycl::event& __e) const
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
               std::size_t __n, const sycl::event& __e) const
    {
        using _GlobalHistKernelT = __global_histogram<__sycl_tag, __is_ascending, __radix_bits, __hist_work_group_count,
                                                      __hist_work_group_size, std::decay_t<_KeysRng>>;

        // Calculate number of histograms based on SLM capacity
        constexpr std::uint32_t __max_histograms = 16;
        constexpr std::uint32_t __max_slm_bytes = 1 << 16;
        constexpr std::uint32_t __num_histograms =
            std::min(__max_histograms,
                     std::uint32_t(__max_slm_bytes / (_GlobalHistKernelT::__hist_buffer_size * sizeof(std::uint32_t))));

        sycl::nd_range<1> __nd_range(__hist_work_group_count * __hist_work_group_size, __hist_work_group_size);
        return __q.submit([&](sycl::handler& __cgh) {
            sycl::local_accessor<std::uint32_t, 1> __slm_accessor(
                _GlobalHistKernelT::__hist_buffer_size * __num_histograms, __cgh);
            oneapi::dpl::__ranges::__require_access(__cgh, __keys_rng);
            __cgh.depends_on(__e);
            _GlobalHistKernelT __kernel(__n, __keys_rng, __slm_accessor, __global_offset_data, __num_histograms);
            __cgh.parallel_for<_Name...>(__nd_range, __kernel);
        });
    }
};

template <std::uint32_t __stage_count, std::uint16_t __bin_count, typename _KernelName>
struct __radix_sort_onesweep_scan_submitter;

template <std::uint32_t __stage_count, std::uint32_t __bin_count, typename... _Name>
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
                std::uint32_t __offset = __nd_item.get_global_id(0);
                const auto __g = __nd_item.get_group();
                std::uint32_t __count = __global_offset_data[__offset];
                std::uint32_t __presum =
                    __dpl_sycl::__exclusive_scan_over_group(__g, __count, __dpl_sycl::__plus<std::uint32_t>());
                __global_offset_data[__offset] = __presum;
            });
        });
    }
};

// We must query info about the kernel in onesweep. To do this, we need to know if a custom name is passed. Since
// we use a struct as a function object, this will be the kernel's name in "unnamed lambda" mode. Otherwise, we use the _CustomName.
template <typename _KernelFuncStruct, typename... _Name>
struct __onesweep_kernel_name_helper;

template <typename _KernelFuncStruct>
struct __onesweep_kernel_name_helper<_KernelFuncStruct,
                                     oneapi::dpl::__par_backend_hetero::__internal::__optional_kernel_name<>>
{
    using kernel_name = _KernelFuncStruct;
};

template <typename _KernelFuncStruct, typename _CustomName>
struct __onesweep_kernel_name_helper<_KernelFuncStruct,
                                     oneapi::dpl::__par_backend_hetero::__internal::__optional_kernel_name<_CustomName>>
{
    using kernel_name = _CustomName;
};

template <bool __is_ascending, std::uint8_t __radix_bits, std::uint16_t __data_per_work_item,
          std::uint16_t __work_group_size, typename _KernelName>
struct __radix_sort_onesweep_submitter;

template <bool __is_ascending, std::uint8_t __radix_bits, std::uint16_t __data_per_work_item,
          std::uint16_t __work_group_size, typename... _Name>
struct __radix_sort_onesweep_submitter<__is_ascending, __radix_bits, __data_per_work_item, __work_group_size,
                                       oneapi::dpl::__par_backend_hetero::__internal::__optional_kernel_name<_Name...>>
{
    template <typename _InRngPack, typename _OutRngPack, typename _GlobalHistT>
    sycl::event
    operator()(__esimd_tag, sycl::queue& __q, _InRngPack&& __in_pack, _OutRngPack&& __out_pack,
               _GlobalHistT* __p_global_hist, _GlobalHistT* __p_group_hists, std::uint32_t __sweep_work_group_count,
               std::size_t __n, std::uint32_t __stage, const sycl::event& __e) const
    {
        sycl::nd_range<1> __nd_range(__sweep_work_group_count * __work_group_size, __work_group_size);
        return __q.submit([&](sycl::handler& __cgh) {
            oneapi::dpl::__ranges::__require_access(__cgh, __in_pack.__keys_rng(), __out_pack.__keys_rng());
            if constexpr (std::decay_t<_InRngPack>::__has_values)
            {
                oneapi::dpl::__ranges::__require_access(__cgh, __in_pack.__vals_rng(), __out_pack.__vals_rng());
            }
            __cgh.depends_on(__e);
            __radix_sort_onesweep_kernel<__esimd_tag, __is_ascending, __radix_bits, __data_per_work_item,
                                         __work_group_size, std::decay_t<_InRngPack>, std::decay_t<_OutRngPack>>
                __kernel(__n, __stage, __p_global_hist, __p_group_hists, std::forward<_InRngPack>(__in_pack),
                         std::forward<_OutRngPack>(__out_pack));
            __cgh.parallel_for<_Name...>(__nd_range, __kernel);
        });
    }

    template <typename _InRngPack, typename _OutRngPack, typename _GlobalHistT>
    sycl::event
    operator()(__sycl_tag, sycl::queue& __q, _InRngPack&& __in_pack, _OutRngPack&& __out_pack,
               _GlobalHistT* __p_global_hist, _GlobalHistT* __p_group_hists, std::uint32_t __sweep_work_group_count,
               std::size_t __n, std::uint32_t __stage, const sycl::event& __e) const
    {
        using _KernelType =
            __radix_sort_onesweep_kernel<__sycl_tag, __is_ascending, __radix_bits, __data_per_work_item,
                                         __work_group_size, std::decay_t<_InRngPack>, std::decay_t<_OutRngPack>>;
        using _KernelName = typename __onesweep_kernel_name_helper<
            _KernelType, oneapi::dpl::__par_backend_hetero::__internal::__optional_kernel_name<_Name...>>::kernel_name;

        // There is a bug produced on BMG where zeKernelSuggestMaxCooperativeGroupCount suggests too large of a
        // work-group count when we are beyond half SLM capacity, causing a hang. To fix this, we can manually compute
        // the safe number of groups to launch and take the min with the root group query for any kernel specific
        // restrictions that may limit the number of groups
        constexpr std::uint32_t __xve_per_xe = 8;
        constexpr std::uint32_t __lanes_per_xe = 2048;
        constexpr std::uint32_t __max_groups_per_xe = __lanes_per_xe / __work_group_size;
        const std::uint32_t __max_slm_xe = __q.get_device().get_info<sycl::info::device::local_mem_size>();
        const std::uint32_t __xes_on_device =
            __q.get_device().get_info<sycl::info::device::max_compute_units>() / __xve_per_xe;

        const std::uint32_t __slm_size_bytes = _KernelType::__calc_slm_alloc();
        const std::uint32_t __groups_per_xe_slm_adj = std::min(__max_groups_per_xe, __max_slm_xe / __slm_size_bytes);

        const std::uint32_t __concurrent_groups_est = __groups_per_xe_slm_adj * __xes_on_device;

        auto __bundle = sycl::get_kernel_bundle<sycl::bundle_state::executable>(__q.get_context());
        auto __kernel = __bundle.template get_kernel<_KernelName>();
        std::uint32_t __max_num_wgs =
            __kernel.template ext_oneapi_get_info<syclex::info::kernel_queue_specific::max_num_work_groups>(
                __q, __work_group_size, __slm_size_bytes);

        std::uint32_t __num_wgs = std::min({__max_num_wgs, __sweep_work_group_count, __concurrent_groups_est});

        sycl::nd_range<1> __nd_range(__num_wgs * __work_group_size, __work_group_size);
        return __q.submit([&](sycl::handler& __cgh) {
            sycl::local_accessor<unsigned char, 1> __slm_accessor(__slm_size_bytes, __cgh);
            oneapi::dpl::__ranges::__require_access(__cgh, __in_pack.__keys_rng(), __out_pack.__keys_rng());
            if constexpr (std::decay_t<_InRngPack>::__has_values)
            {
                oneapi::dpl::__ranges::__require_access(__cgh, __in_pack.__vals_rng(), __out_pack.__vals_rng());
            }
            __cgh.depends_on(__e);
            _KernelType __kernel(__n, __stage, __p_global_hist, __p_group_hists, std::forward<_InRngPack>(__in_pack),
                                 std::forward<_OutRngPack>(__out_pack), __slm_accessor, __sweep_work_group_count);
            __cgh.parallel_for<_KernelName>(__nd_range, __kernel);
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
    operator()(_KtTag, sycl::queue& __q, _InRngPack&& __in_pack, _OutRngPack&& __out_pack, std::uint32_t __n,
               const sycl::event& __e) const
    {
        // Copyback kernel is pure sycl for esimd_sort, so no need to dispatch from tag
        return __q.submit([&](sycl::handler& __cgh) {
            oneapi::dpl::__ranges::__require_access(__cgh, __in_pack.__keys_rng(), __out_pack.__keys_rng());
            if constexpr (std::decay_t<_InRngPack>::__has_values)
            {
                oneapi::dpl::__ranges::__require_access(__cgh, __in_pack.__vals_rng(), __out_pack.__vals_rng());
            }
            // TODO: make sure that access is read_only for __keys_tmp_rng/__vals_tmp_rng  and is write_only for __keys_rng/__vals_rng
            __cgh.depends_on(__e);
            __cgh.parallel_for<_Name...>(sycl::range<1>{__n}, [=](sycl::item<1> __item) {
                auto __global_id = __item.get_linear_id();
                __rng_data(__out_pack.__keys_rng())[__global_id] = __rng_data(__in_pack.__keys_rng())[__global_id];
                if constexpr (std::decay_t<_InRngPack>::__has_values)
                {
                    __rng_data(__out_pack.__vals_rng())[__global_id] = __rng_data(__in_pack.__vals_rng())[__global_id];
                }
            });
        });
    }
};

} // namespace oneapi::dpl::experimental::kt::gpu::__impl

#endif // _ONEDPL_KT_RADIX_SORT_SUBMITTERS_H
