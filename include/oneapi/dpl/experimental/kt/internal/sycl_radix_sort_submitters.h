// -*- C++ -*-
//===-- sycl_radix_sort_submitters.h -------------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _ONEDPL_KT_SYCL_RADIX_SORT_SUBMITTERS_H
#define _ONEDPL_KT_SYCL_RADIX_SORT_SUBMITTERS_H

#include <cstdint>
#include <utility>
#include <type_traits>

#include "../../../pstl/hetero/dpcpp/sycl_defs.h"
#include "../../../pstl/hetero/dpcpp/utils_ranges_sycl.h"
#include "../../../pstl/hetero/dpcpp/parallel_backend_sycl_utils.h"

#include "sycl_radix_sort_kernels.h"
#include "sycl_radix_sort_utils.h"

// TEMPORARY: Include ESIMD implementations for baseline
#include "esimd_radix_sort_kernels.h"

namespace oneapi::dpl::experimental::kt::gpu::__sycl::__impl
{

//-----------------------------------------------------------------------------
// Kernel name tags
//-----------------------------------------------------------------------------
template <typename... _Name>
class __sycl_radix_sort_onesweep_histogram;

template <typename... _Name>
class __sycl_radix_sort_onesweep_scan;

template <typename... _Name>
class __sycl_radix_sort_onesweep;

template <typename... _Name>
class __sycl_radix_sort_onesweep_by_key;

template <typename... _Name>
class __sycl_radix_sort_onesweep_copyback;

template <typename... _Name>
class __sycl_radix_sort_onesweep_copyback_by_key;

//-----------------------------------------------------------------------------
// Histogram submitter
//-----------------------------------------------------------------------------
template <bool __is_ascending, std::uint8_t __radix_bits,
          std::uint32_t __hist_work_group_count, std::uint16_t __hist_work_group_size,
          typename _KernelName>
struct __radix_sort_histogram_submitter;

template <bool __is_ascending, std::uint8_t __radix_bits,
          std::uint32_t __hist_work_group_count, std::uint16_t __hist_work_group_size,
          typename... _Name>
struct __radix_sort_histogram_submitter<
    __is_ascending, __radix_bits, __hist_work_group_count, __hist_work_group_size,
    oneapi::dpl::__par_backend_hetero::__internal::__optional_kernel_name<_Name...>>
{
    template <typename _KeysRng, typename _GlobalHistData>
    sycl::event
    operator()(sycl::queue& __q, const _KeysRng& __keys_rng,
               const _GlobalHistData& __global_hist_data, std::size_t __n,
               const sycl::event& __e) const
    {
        // TEMPORARY: Use ESIMD implementation as baseline
        using namespace oneapi::dpl::experimental::kt::gpu::esimd::__impl;
        sycl::nd_range<1> __nd_range(__hist_work_group_count * __hist_work_group_size,
                                     __hist_work_group_size);
        return __q.submit([&](sycl::handler& __cgh) {
            oneapi::dpl::__ranges::__require_access(__cgh, __keys_rng);
            __cgh.depends_on(__e);
            __cgh.parallel_for<_Name...>(__nd_range, [=](sycl::nd_item<1> __nd_item) [[intel::sycl_explicit_simd]] {
                __global_histogram<__is_ascending, __radix_bits, __hist_work_group_count, __hist_work_group_size>(
                    __nd_item, __n, __keys_rng, __global_hist_data);
            });
        });
    }
};

//-----------------------------------------------------------------------------
// Scan submitter (scans global histogram to produce offsets)
//-----------------------------------------------------------------------------
template <std::uint32_t __stage_count, std::uint16_t __bin_count, typename _KernelName>
struct __radix_sort_onesweep_scan_submitter;

template <std::uint32_t __stage_count, std::uint32_t __bin_count, typename... _Name>
struct __radix_sort_onesweep_scan_submitter<
    __stage_count, __bin_count,
    oneapi::dpl::__par_backend_hetero::__internal::__optional_kernel_name<_Name...>>
{
    template <typename _GlobalOffsetData>
    sycl::event
    operator()(sycl::queue& __q, const _GlobalOffsetData& __global_offset_data,
               const sycl::event& __e) const
    {
        // Use SYCL group scan
        sycl::nd_range<1> __nd_range(__stage_count * __bin_count, __bin_count);
        return __q.submit([&](sycl::handler& __cgh) {
            __cgh.depends_on(__e);
            __cgh.parallel_for<_Name...>(__nd_range, [=](sycl::nd_item<1> __item) {
                std::uint32_t __offset = __item.get_global_id(0);
                const auto __g = __item.get_group();
                std::uint32_t __count = __global_offset_data[__offset];
                std::uint32_t __presum = sycl::exclusive_scan_over_group(
                    __g, __count, sycl::plus<std::uint32_t>());
                __global_offset_data[__offset] = __presum;
            });
        });
    }
};

//-----------------------------------------------------------------------------
// Onesweep submitter (main sweep/reorder kernel)
//-----------------------------------------------------------------------------
template <bool __is_ascending, std::uint8_t __radix_bits,
          std::uint16_t __data_per_work_item, std::uint16_t __work_group_size,
          typename _KernelName>
struct __radix_sort_onesweep_submitter;

template <bool __is_ascending, std::uint8_t __radix_bits,
          std::uint16_t __data_per_work_item, std::uint16_t __work_group_size,
          typename... _Name>
struct __radix_sort_onesweep_submitter<
    __is_ascending, __radix_bits, __data_per_work_item, __work_group_size,
    oneapi::dpl::__par_backend_hetero::__internal::__optional_kernel_name<_Name...>>
{
    template <typename _InRngPack, typename _OutRngPack, typename _GlobalHistT>
    sycl::event
    operator()(sycl::queue& __q, _InRngPack&& __in_pack, _OutRngPack&& __out_pack,
               _GlobalHistT* __p_global_hist, _GlobalHistT* __p_group_hists,
               std::uint32_t __sweep_work_group_count, std::size_t __n,
               std::uint32_t __stage, const sycl::event& __e) const
    {
        // TEMPORARY: Use ESIMD implementation as baseline
        using namespace oneapi::dpl::experimental::kt::gpu::esimd::__impl;
        sycl::nd_range<1> __nd_range(__sweep_work_group_count * __work_group_size,
                                     __work_group_size);
        return __q.submit([&](sycl::handler& __cgh) {
            oneapi::dpl::__ranges::__require_access(__cgh, __in_pack.__keys_rng(),
                                                    __out_pack.__keys_rng());
            if constexpr (std::decay_t<_InRngPack>::__has_values)
            {
                oneapi::dpl::__ranges::__require_access(__cgh, __in_pack.__vals_rng(),
                                                        __out_pack.__vals_rng());
            }
            __cgh.depends_on(__e);

            __radix_sort_onesweep_kernel<__is_ascending, __radix_bits, __data_per_work_item, __work_group_size,
                                         std::decay_t<_InRngPack>, std::decay_t<_OutRngPack>>
                __kernel(__n, __stage, __p_global_hist, __p_group_hists,
                         std::forward<_InRngPack>(__in_pack), std::forward<_OutRngPack>(__out_pack));
            __cgh.parallel_for<_Name...>(__nd_range, __kernel);
        });
    }
};

//-----------------------------------------------------------------------------
// Copyback submitter (for in-place with odd stages)
//-----------------------------------------------------------------------------
template <typename _KernelName>
struct __radix_sort_copyback_submitter;

template <typename... _Name>
struct __radix_sort_copyback_submitter<
    oneapi::dpl::__par_backend_hetero::__internal::__optional_kernel_name<_Name...>>
{
    template <typename _InRngPack, typename _OutRngPack>
    sycl::event
    operator()(sycl::queue& __q, _InRngPack&& __in_pack, _OutRngPack&& __out_pack,
               std::uint32_t __n, const sycl::event& __e) const
    {
        // Simple copy kernel
        constexpr std::size_t __work_group_size = 256;
        const std::size_t __global_size =
            (((__n + __work_group_size - 1) / __work_group_size) * __work_group_size);

        return __q.submit([&](sycl::handler& __cgh) {
            oneapi::dpl::__ranges::__require_access(__cgh, __in_pack.__keys_rng(),
                                                    __out_pack.__keys_rng());
            if constexpr (std::decay_t<_InRngPack>::__has_values)
            {
                oneapi::dpl::__ranges::__require_access(__cgh, __in_pack.__vals_rng(),
                                                        __out_pack.__vals_rng());
            }
            __cgh.depends_on(__e);

            __cgh.parallel_for<_Name...>(
                sycl::nd_range<1>(__global_size, __work_group_size),
                [=](sycl::nd_item<1> __item) {
                    std::uint32_t __idx = __item.get_global_id(0);
                    if (__idx < __n)
                    {
                        auto __in_keys = __rng_data(__in_pack.__keys_rng());
                        auto __out_keys = __rng_data(__out_pack.__keys_rng());
                        __out_keys[__idx] = __in_keys[__idx];

                        if constexpr (std::decay_t<_InRngPack>::__has_values)
                        {
                            auto __in_vals = __rng_data(__in_pack.__vals_rng());
                            auto __out_vals = __rng_data(__out_pack.__vals_rng());
                            __out_vals[__idx] = __in_vals[__idx];
                        }
                    }
                });
        });
    }
};

} // namespace oneapi::dpl::experimental::kt::gpu::__sycl::__impl

#endif // _ONEDPL_KT_SYCL_RADIX_SORT_SUBMITTERS_H
