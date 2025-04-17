// -*- C++ -*-
//===-- esimd_radix_sort.h --------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===------------------------------------------------------===//

#ifndef _ONEDPL_KT_ESIMD_RADIX_SORT_H
#define _ONEDPL_KT_ESIMD_RADIX_SORT_H

#include <cstdint>
#include <type_traits>

#include "../../pstl/hetero/dpcpp/utils_ranges_sycl.h"

#include "internal/esimd_radix_sort_utils.h"
#include "internal/esimd_radix_sort_dispatchers.h"
#include "../../pstl/utils.h"

namespace oneapi::dpl::experimental::kt::gpu::esimd
{

// TODO: make sure to provide sufficient diagnostic if input does not allow either reading or writing
template <bool __is_ascending = true, ::std::uint8_t __radix_bits = 8, typename _KernelParam, typename _KeysRng>
std::enable_if_t<!oneapi::dpl::__internal::__is_iterator_type_v<_KeysRng>, sycl::event>
radix_sort(sycl::queue __q, _KeysRng&& __keys_rng, _KernelParam __param = {})
{
    __impl::__check_esimd_sort_params<__radix_bits, _KernelParam::data_per_workitem, _KernelParam::workgroup_size>();

    if (__keys_rng.size() < 2)
        return {};

    auto __pack = __impl::__rng_pack{oneapi::dpl::__ranges::views::all(::std::forward<_KeysRng>(__keys_rng))};
    return __impl::__radix_sort<__is_ascending, __radix_bits, /*__in_place=*/true>(__q, __pack, __pack, __param);
}

template <bool __is_ascending = true, ::std::uint8_t __radix_bits = 8, typename _KernelParam, typename _KeysIterator>
std::enable_if_t<oneapi::dpl::__internal::__is_iterator_type_v<_KeysIterator>, sycl::event>
radix_sort(sycl::queue __q, _KeysIterator __keys_first, _KeysIterator __keys_last, _KernelParam __param = {})
{
    __impl::__check_esimd_sort_params<__radix_bits, _KernelParam::data_per_workitem, _KernelParam::workgroup_size>();

    if (__keys_last - __keys_first < 2)
        return {};

    auto __keys_keep = oneapi::dpl::__ranges::__get_sycl_range<sycl::access_mode::read_write, _KeysIterator>();
    auto __keys_rng = __keys_keep(__keys_first, __keys_last).all_view();
    auto __pack = __impl::__rng_pack{::std::move(__keys_rng)};
    return __impl::__radix_sort<__is_ascending, __radix_bits, /*__in_place=*/true>(__q, __pack, __pack, __param);
}

template <bool __is_ascending = true, ::std::uint8_t __radix_bits = 8, typename _KernelParam, typename _KeysRng,
          typename _ValsRng>
std::enable_if_t<!oneapi::dpl::__internal::__is_iterator_type_v<_KeysRng>, sycl::event>
radix_sort_by_key(sycl::queue __q, _KeysRng&& __keys_rng, _ValsRng&& __vals_rng, _KernelParam __param = {})
{
    __impl::__check_esimd_sort_params<__radix_bits, _KernelParam::data_per_workitem, _KernelParam::workgroup_size>();

    if (__keys_rng.size() < 2)
        return {};

    auto __pack = __impl::__rng_pack{oneapi::dpl::__ranges::views::all(::std::forward<_KeysRng>(__keys_rng)),
                                     oneapi::dpl::__ranges::views::all(::std::forward<_ValsRng>(__vals_rng))};
    return __impl::__radix_sort<__is_ascending, __radix_bits, /*__in_place=*/true>(__q, __pack, __pack, __param);
}

template <bool __is_ascending = true, ::std::uint8_t __radix_bits = 8, typename _KernelParam, typename _KeysIterator,
          typename _ValsIterator>
std::enable_if_t<oneapi::dpl::__internal::__is_iterator_type_v<_KeysIterator>, sycl::event>
radix_sort_by_key(sycl::queue __q, _KeysIterator __keys_first, _KeysIterator __keys_last, _ValsIterator __vals_first,
                  _KernelParam __param = {})
{
    __impl::__check_esimd_sort_params<__radix_bits, _KernelParam::data_per_workitem, _KernelParam::workgroup_size>();

    if (__keys_last - __keys_first < 2)
        return {};

    auto __keys_keep = oneapi::dpl::__ranges::__get_sycl_range<sycl::access_mode::read_write, _KeysIterator>();
    auto __keys_rng = __keys_keep(__keys_first, __keys_last).all_view();

    auto __vals_keep = oneapi::dpl::__ranges::__get_sycl_range<sycl::access_mode::read_write, _ValsIterator>();
    auto __vals_rng = __vals_keep(__vals_first, __vals_first + (__keys_last - __keys_first)).all_view();
    auto __pack = __impl::__rng_pack{::std::move(__keys_rng), ::std::move(__vals_rng)};
    return __impl::__radix_sort<__is_ascending, __radix_bits, /*__in_place=*/true>(__q, __pack, __pack, __param);
}

template <bool __is_ascending = true, ::std::uint8_t __radix_bits = 8, typename _KernelParam, typename _KeysRng1,
          typename _KeysRng2>
std::enable_if_t<!oneapi::dpl::__internal::__is_iterator_type_v<_KeysRng1>, sycl::event>
radix_sort(sycl::queue __q, _KeysRng1&& __keys_rng, _KeysRng2&& __keys_rng_out, _KernelParam __param = {})
{
    __impl::__check_esimd_sort_params<__radix_bits, _KernelParam::data_per_workitem, _KernelParam::workgroup_size>();
    if (__keys_rng.size() == 0)
        return {};

    auto __pack = __impl::__rng_pack{oneapi::dpl::__ranges::views::all(::std::forward<_KeysRng1>(__keys_rng))};
    auto __pack_out = __impl::__rng_pack{oneapi::dpl::__ranges::views::all(::std::forward<_KeysRng2>(__keys_rng_out))};
    return __impl::__radix_sort<__is_ascending, __radix_bits, /*__in_place=*/false>(__q, ::std::move(__pack),
                                                                                    ::std::move(__pack_out), __param);
}

template <bool __is_ascending = true, ::std::uint8_t __radix_bits = 8, typename _KernelParam, typename _KeysIterator1,
          typename _KeysIterator2>
std::enable_if_t<oneapi::dpl::__internal::__is_iterator_type_v<_KeysIterator1>, sycl::event>
radix_sort(sycl::queue __q, _KeysIterator1 __keys_first, _KeysIterator1 __keys_last, _KeysIterator2 __keys_out_first,
           _KernelParam __param = {})
{
    __impl::__check_esimd_sort_params<__radix_bits, _KernelParam::data_per_workitem, _KernelParam::workgroup_size>();

    auto __n = __keys_last - __keys_first;
    if (__n == 0)
        return {};

    auto __keys_keep = oneapi::dpl::__ranges::__get_sycl_range<sycl::access_mode::read_write, _KeysIterator1>();
    auto __keys_rng = __keys_keep(__keys_first, __keys_last).all_view();
    auto __pack = __impl::__rng_pack{::std::move(__keys_rng)};
    auto __keys_out_keep = oneapi::dpl::__ranges::__get_sycl_range<sycl::access_mode::read_write, _KeysIterator2>();
    auto __keys_out_rng = __keys_out_keep(__keys_out_first, __keys_out_first + __n).all_view();
    auto __pack_out = __impl::__rng_pack{::std::move(__keys_out_rng)};
    return __impl::__radix_sort<__is_ascending, __radix_bits, /*__in_place=*/false>(__q, ::std::move(__pack),
                                                                                    ::std::move(__pack_out), __param);
}

template <bool __is_ascending = true, ::std::uint8_t __radix_bits = 8, typename _KernelParam, typename _KeysRng1,
          typename _ValsRng1, typename _KeysRng2, typename _ValsRng2>
std::enable_if_t<!oneapi::dpl::__internal::__is_iterator_type_v<_KeysRng1>, sycl::event>
radix_sort_by_key(sycl::queue __q, _KeysRng1&& __keys_rng, _ValsRng1&& __vals_rng, _KeysRng2&& __keys_out_rng,
                  _ValsRng2&& __vals_out_rng, _KernelParam __param = {})
{
    __impl::__check_esimd_sort_params<__radix_bits, _KernelParam::data_per_workitem, _KernelParam::workgroup_size>();
    if (__keys_rng.size() == 0)
        return {};

    auto __pack = __impl::__rng_pack{oneapi::dpl::__ranges::views::all(::std::forward<_KeysRng1>(__keys_rng)),
                                     oneapi::dpl::__ranges::views::all(::std::forward<_ValsRng1>(__vals_rng))};
    auto __pack_out = __impl::__rng_pack{oneapi::dpl::__ranges::views::all(::std::forward<_KeysRng2>(__keys_out_rng)),
                                         oneapi::dpl::__ranges::views::all(::std::forward<_ValsRng2>(__vals_out_rng))};
    return __impl::__radix_sort<__is_ascending, __radix_bits, /*__in_place=*/false>(__q, ::std::move(__pack),
                                                                                    ::std::move(__pack_out), __param);
}

template <bool __is_ascending = true, ::std::uint8_t __radix_bits = 8, typename _KernelParam, typename _KeysIterator1,
          typename _ValsIterator1, typename _KeysIterator2, typename _ValsIterator2>
std::enable_if_t<oneapi::dpl::__internal::__is_iterator_type_v<_KeysIterator1>, sycl::event>
radix_sort_by_key(sycl::queue __q, _KeysIterator1 __keys_first, _KeysIterator1 __keys_last, _ValsIterator1 __vals_first,
                  _KeysIterator2 __keys_out_first, _ValsIterator2 __vals_out_first, _KernelParam __param = {})
{
    __impl::__check_esimd_sort_params<__radix_bits, _KernelParam::data_per_workitem, _KernelParam::workgroup_size>();

    auto __n = __keys_last - __keys_first;
    if (__n == 0)
        return {};

    auto __keys_keep = oneapi::dpl::__ranges::__get_sycl_range<sycl::access_mode::read_write, _KeysIterator1>();
    auto __keys_rng = __keys_keep(__keys_first, __keys_last).all_view();

    auto __vals_keep = oneapi::dpl::__ranges::__get_sycl_range<sycl::access_mode::read_write, _ValsIterator1>();
    auto __vals_rng = __vals_keep(__vals_first, __vals_first + __n).all_view();
    auto __pack = __impl::__rng_pack{::std::move(__keys_rng), ::std::move(__vals_rng)};

    auto __keys_out_keep = oneapi::dpl::__ranges::__get_sycl_range<sycl::access_mode::read_write, _KeysIterator2>();
    auto __keys_out_rng = __keys_keep(__keys_out_first, __keys_out_first + __n).all_view();

    auto __vals_out_keep = oneapi::dpl::__ranges::__get_sycl_range<sycl::access_mode::read_write, _ValsIterator2>();
    auto __vals_out_rng = __vals_keep(__vals_out_first, __vals_out_first + __n).all_view();
    auto __pack_out = __impl::__rng_pack{::std::move(__keys_out_rng), ::std::move(__vals_out_rng)};
    return __impl::__radix_sort<__is_ascending, __radix_bits, /*__in_place=*/false>(__q, ::std::move(__pack),
                                                                                    ::std::move(__pack_out), __param);
}

} // namespace oneapi::dpl::experimental::kt::gpu::esimd

namespace oneapi::dpl::experimental::kt
{
namespace
#if !defined(__SYCL_DEVICE_ONLY__)
    [[deprecated("Use of oneapi::dpl::experimental::kt::esimd namespace is deprecated "
                 "and will be removed in a future release. "
                 "Use oneapi::dpl::experimental::kt::gpu::esimd instead")]]
#endif
    esimd
{
using oneapi::dpl::experimental::kt::gpu::esimd::radix_sort;
using oneapi::dpl::experimental::kt::gpu::esimd::radix_sort_by_key;
} // namespace oneapi::dpl::experimental::kt::esimd
} // namespace oneapi::dpl::experimental::kt

#endif // _ONEDPL_KT_ESIMD_RADIX_SORT_H
