// -*- C++ -*-
//===-- esimd_radix_sort_submitters.h --------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===------------------------------------------------------===//

#ifndef _ONEDPL_KT_ESIMD_RADIX_SORT_SUBMITTERS_H
#define _ONEDPL_KT_ESIMD_RADIX_SORT_SUBMITTERS_H

#include <cstdint>
#include <utility>
#include <type_traits>

#include "../../../pstl/hetero/dpcpp/sycl_defs.h"

#include "../../../pstl/hetero/dpcpp/utils_ranges_sycl.h"
#include "../../../pstl/hetero/dpcpp/parallel_backend_sycl_utils.h"
#include "../../../pstl/hetero/dpcpp/sycl_traits.h" //SYCL traits specialization for some oneDPL types.

#include "esimd_radix_sort_kernels.h"
#include "esimd_defs.h"

namespace oneapi::dpl::experimental::kt::gpu::esimd::__impl
{

//------------------------------------------------------------------------
// Please see the comment above __parallel_for_small_submitter for optional kernel name explanation
//------------------------------------------------------------------------

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
    operator()(sycl::queue __q, _RngPack1&& __pack_in, _RngPack2&& __pack_out, ::std::size_t __n) const
    {
        sycl::nd_range<1> __nd_range{__work_group_size, __work_group_size};
        return __q.submit([&](sycl::handler& __cgh) {
            oneapi::dpl::__ranges::__require_access(__cgh, __pack_in.__keys_rng());
            oneapi::dpl::__ranges::__require_access(__cgh, __pack_out.__keys_rng());
            __cgh.parallel_for<_Name...>(__nd_range, [=](sycl::nd_item<1> __nd_item) [[intel::sycl_explicit_simd]] {
                __one_wg_kernel<__is_ascending, __radix_bits, __data_per_work_item, __work_group_size, _KeyT>(
                    __nd_item, __n, __pack_in, __pack_out);
            });
        });
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
    operator()(sycl::queue& __q, const _KeysRng& __keys_rng, const _GlobalOffsetData& __global_offset_data,
               ::std::size_t __n, const sycl::event& __e) const
    {
        sycl::nd_range<1> __nd_range(__hist_work_group_count * __hist_work_group_size, __hist_work_group_size);
        return __q.submit([&](sycl::handler& __cgh) {
            oneapi::dpl::__ranges::__require_access(__cgh, __keys_rng);
            __cgh.depends_on(__e);
            __cgh.parallel_for<_Name...>(__nd_range, [=](sycl::nd_item<1> __nd_item) [[intel::sycl_explicit_simd]] {
                __global_histogram<__is_ascending, __radix_bits, __hist_work_group_count, __hist_work_group_size>(
                    __nd_item, __n, __keys_rng, __global_offset_data);
            });
        });
    }
};

template <::std::uint32_t __stage_count, ::std::uint16_t __bin_count, typename _KernelName>
struct __radix_sort_onesweep_scan_submitter;

template <::std::uint32_t __stage_count, ::std::uint32_t __bin_count, typename... _Name>
struct __radix_sort_onesweep_scan_submitter<
    __stage_count, __bin_count, oneapi::dpl::__par_backend_hetero::__internal::__optional_kernel_name<_Name...>>
{
    template <typename _GlobalOffsetData>
    sycl::event
    operator()(sycl::queue& __q, const _GlobalOffsetData& __global_offset_data, ::std::size_t __n,
               const sycl::event& __e) const
    {
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
    template <typename _InRngPack, typename _OutRngPack, typename _GlobalHistT>
    sycl::event
    operator()(sycl::queue& __q, _InRngPack&& __in_pack, _OutRngPack&& __out_pack, _GlobalHistT* __p_global_hist,
               _GlobalHistT* __p_group_hists, ::std::uint32_t __sweep_work_group_count, ::std::size_t __n,
               ::std::uint32_t __stage, const sycl::event& __e) const
    {
        sycl::nd_range<1> __nd_range(__sweep_work_group_count * __work_group_size, __work_group_size);
        return __q.submit([&](sycl::handler& __cgh) {
            oneapi::dpl::__ranges::__require_access(__cgh, __in_pack.__keys_rng(), __out_pack.__keys_rng());
            if constexpr (::std::decay_t<_InRngPack>::__has_values)
            {
                oneapi::dpl::__ranges::__require_access(__cgh, __in_pack.__vals_rng(), __out_pack.__vals_rng());
            }
            __cgh.depends_on(__e);
            __radix_sort_onesweep_kernel<__is_ascending, __radix_bits, __data_per_work_item, __work_group_size,
                                         ::std::decay_t<_InRngPack>, ::std::decay_t<_OutRngPack>>
                __kernel(__n, __stage, __p_global_hist, __p_group_hists, ::std::forward<_InRngPack>(__in_pack),
                         ::std::forward<_OutRngPack>(__out_pack));
            __cgh.parallel_for<_Name...>(__nd_range, __kernel);
        });
    }
};

template <typename _KernelName>
struct __radix_sort_copyback_submitter;

template <typename... _Name>
struct __radix_sort_copyback_submitter<oneapi::dpl::__par_backend_hetero::__internal::__optional_kernel_name<_Name...>>
{
    template <typename _InRngPack, typename _OutRngPack>
    sycl::event
    operator()(sycl::queue& __q, _InRngPack&& __in_pack, _OutRngPack&& __out_pack, ::std::uint32_t __n,
               const sycl::event& __e) const
    {
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

} // namespace oneapi::dpl::experimental::kt::gpu::esimd::__impl

#endif // _ONEDPL_KT_ESIMD_RADIX_SORT_SUBMITTERS_H
