// -*- C++ -*-
//===-- batched_merge_sort.h ----------------------------------------------===//
//
// Copyright (C) UXL Foundation Contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _ONEDPL_KT_BATCHED_MERGE_SORT_H
#define _ONEDPL_KT_BATCHED_MERGE_SORT_H

#include <cstdint>
#include <cstddef>
#include <type_traits>
#include <utility>

#include "../../pstl/hetero/dpcpp/utils_ranges_sycl.h"
#include "internal/radix_sort_utils.h"
#include "internal/batched_merge_sort_kernels.h"

namespace oneapi::dpl::experimental::kt::gpu
{
template <bool __is_ascending = true, typename _KernelParam, typename _KeysRng>
std::enable_if_t<!oneapi::dpl::__internal::__is_type_with_iterator_traits_v<_KeysRng>, sycl::event>
batched_merge_sort(sycl::queue __q, _KeysRng&& __keys_rng, std::size_t __segment_size, _KernelParam __param = {})
{
    auto __n = __keys_rng.size();
    __impl::__check_batched_merge_sort_params<_KernelParam::data_per_workitem, _KernelParam::workgroup_size>(
        __n, __segment_size);
    if (__n == 0)
        return {};

    auto __pack = __impl::__rng_pack{oneapi::dpl::__ranges::views::all(std::forward<_KeysRng>(__keys_rng))};
    return __impl::__batched_merge_sort<__is_ascending>(__q, __pack, __pack, __segment_size, __param);
}

template <bool __is_ascending = true, typename _KernelParam, typename _KeysIterator>
std::enable_if_t<oneapi::dpl::__internal::__is_type_with_iterator_traits_v<_KeysIterator>, sycl::event>
batched_merge_sort(sycl::queue __q, _KeysIterator __keys_first, _KeysIterator __keys_last, std::size_t __segment_size,
                   _KernelParam __param = {})
{
    auto __n = __keys_last - __keys_first;
    __impl::__check_batched_merge_sort_params<_KernelParam::data_per_workitem, _KernelParam::workgroup_size>(
        __n, __segment_size);
    if (__n == 0)
        return {};

    auto __keys_keep = oneapi::dpl::__ranges::__get_sycl_range<sycl::access_mode::read_write>();
    auto __keys_rng = __keys_keep(__keys_first, __keys_last).all_view();
    auto __pack = __impl::__rng_pack{std::move(__keys_rng)};
    return __impl::__batched_merge_sort<__is_ascending>(__q, __pack, __pack, __segment_size, __param);
}

template <bool __is_ascending = true, typename _KernelParam, typename _KeysRng, typename _ValsRng>
std::enable_if_t<!oneapi::dpl::__internal::__is_type_with_iterator_traits_v<_KeysRng>, sycl::event>
batched_merge_sort_by_key(sycl::queue __q, _KeysRng&& __keys_rng, _ValsRng&& __vals_rng, std::size_t __segment_size,
                          _KernelParam __param = {})
{
    auto __n = __keys_rng.size();
    __impl::__check_batched_merge_sort_params<_KernelParam::data_per_workitem, _KernelParam::workgroup_size>(
        __n, __segment_size);
    if (__n == 0)
        return {};

    auto __pack = __impl::__rng_pack{oneapi::dpl::__ranges::views::all(std::forward<_KeysRng>(__keys_rng)),
                                     oneapi::dpl::__ranges::views::all(std::forward<_ValsRng>(__vals_rng))};
    return __impl::__batched_merge_sort<__is_ascending>(__q, __pack, __pack, __segment_size, __param);
}

template <bool __is_ascending = true, typename _KernelParam, typename _KeysIterator, typename _ValsIterator>
std::enable_if_t<oneapi::dpl::__internal::__is_type_with_iterator_traits_v<_KeysIterator>, sycl::event>
batched_merge_sort_by_key(sycl::queue __q, _KeysIterator __keys_first, _KeysIterator __keys_last,
                          _ValsIterator __vals_first, std::size_t __segment_size, _KernelParam __param = {})
{
    auto __n = __keys_last - __keys_first;
    __impl::__check_batched_merge_sort_params<_KernelParam::data_per_workitem, _KernelParam::workgroup_size>(
        __n, __segment_size);
    if (__n == 0)
        return {};

    auto __keys_keep = oneapi::dpl::__ranges::__get_sycl_range<sycl::access_mode::read_write>();
    auto __keys_rng = __keys_keep(__keys_first, __keys_last).all_view();
    auto __vals_keep = oneapi::dpl::__ranges::__get_sycl_range<sycl::access_mode::read_write>();
    auto __vals_rng = __vals_keep(__vals_first, __vals_first + __n).all_view();
    auto __pack = __impl::__rng_pack{std::move(__keys_rng), std::move(__vals_rng)};
    return __impl::__batched_merge_sort<__is_ascending>(__q, __pack, __pack, __segment_size, __param);
}

template <bool __is_ascending = true, typename _KernelParam, typename _KeysRng1, typename _KeysRng2>
std::enable_if_t<!oneapi::dpl::__internal::__is_type_with_iterator_traits_v<_KeysRng1>, sycl::event>
batched_merge_sort(sycl::queue __q, _KeysRng1&& __keys_rng, _KeysRng2&& __keys_out_rng, std::size_t __segment_size,
                   _KernelParam __param = {})
{
    auto __n = __keys_rng.size();
    __impl::__check_batched_merge_sort_params<_KernelParam::data_per_workitem, _KernelParam::workgroup_size>(
        __n, __segment_size);
    if (__n == 0)
        return {};

    auto __pack = __impl::__rng_pack{oneapi::dpl::__ranges::views::all(std::forward<_KeysRng1>(__keys_rng))};
    auto __pack_out = __impl::__rng_pack{oneapi::dpl::__ranges::views::all(std::forward<_KeysRng2>(__keys_out_rng))};
    return __impl::__batched_merge_sort<__is_ascending>(__q, std::move(__pack), std::move(__pack_out), __segment_size,
                                                        __param);
}

template <bool __is_ascending = true, typename _KernelParam, typename _KeysIterator1, typename _KeysIterator2>
std::enable_if_t<oneapi::dpl::__internal::__is_type_with_iterator_traits_v<_KeysIterator1>, sycl::event>
batched_merge_sort(sycl::queue __q, _KeysIterator1 __keys_first, _KeysIterator1 __keys_last,
                   _KeysIterator2 __keys_out_first, std::size_t __segment_size, _KernelParam __param = {})
{
    auto __n = __keys_last - __keys_first;
    __impl::__check_batched_merge_sort_params<_KernelParam::data_per_workitem, _KernelParam::workgroup_size>(
        __n, __segment_size);
    if (__n == 0)
        return {};

    auto __keys_keep = oneapi::dpl::__ranges::__get_sycl_range<sycl::access_mode::read_write>();
    auto __keys_rng = __keys_keep(__keys_first, __keys_last).all_view();
    auto __pack = __impl::__rng_pack{std::move(__keys_rng)};
    auto __keys_out_keep = oneapi::dpl::__ranges::__get_sycl_range<sycl::access_mode::read_write>();
    auto __keys_out_rng = __keys_out_keep(__keys_out_first, __keys_out_first + __n).all_view();
    auto __pack_out = __impl::__rng_pack{std::move(__keys_out_rng)};
    return __impl::__batched_merge_sort<__is_ascending>(__q, std::move(__pack), std::move(__pack_out), __segment_size,
                                                        __param);
}

template <bool __is_ascending = true, typename _KernelParam, typename _KeysRng1, typename _ValsRng1, typename _KeysRng2,
          typename _ValsRng2>
std::enable_if_t<!oneapi::dpl::__internal::__is_type_with_iterator_traits_v<_KeysRng1>, sycl::event>
batched_merge_sort_by_key(sycl::queue __q, _KeysRng1&& __keys_rng, _ValsRng1&& __vals_rng, _KeysRng2&& __keys_out_rng,
                          _ValsRng2&& __vals_out_rng, std::size_t __segment_size, _KernelParam __param = {})
{
    auto __n = __keys_rng.size();
    __impl::__check_batched_merge_sort_params<_KernelParam::data_per_workitem, _KernelParam::workgroup_size>(
        __n, __segment_size);
    if (__n == 0)
        return {};

    auto __pack = __impl::__rng_pack{oneapi::dpl::__ranges::views::all(std::forward<_KeysRng1>(__keys_rng)),
                                     oneapi::dpl::__ranges::views::all(std::forward<_ValsRng1>(__vals_rng))};
    auto __pack_out = __impl::__rng_pack{oneapi::dpl::__ranges::views::all(std::forward<_KeysRng2>(__keys_out_rng)),
                                         oneapi::dpl::__ranges::views::all(std::forward<_ValsRng2>(__vals_out_rng))};
    return __impl::__batched_merge_sort<__is_ascending>(__q, std::move(__pack), std::move(__pack_out), __segment_size,
                                                        __param);
}

template <bool __is_ascending = true, typename _KernelParam, typename _KeysIterator1, typename _ValsIterator1,
          typename _KeysIterator2, typename _ValsIterator2>
std::enable_if_t<oneapi::dpl::__internal::__is_type_with_iterator_traits_v<_KeysIterator1>, sycl::event>
batched_merge_sort_by_key(sycl::queue __q, _KeysIterator1 __keys_first, _KeysIterator1 __keys_last,
                          _ValsIterator1 __vals_first, _KeysIterator2 __keys_out_first, _ValsIterator2 __vals_out_first,
                          std::size_t __segment_size, _KernelParam __param = {})
{
    auto __n = __keys_last - __keys_first;
    __impl::__check_batched_merge_sort_params<_KernelParam::data_per_workitem, _KernelParam::workgroup_size>(
        __n, __segment_size);
    if (__n == 0)
        return {};

    auto __keys_keep = oneapi::dpl::__ranges::__get_sycl_range<sycl::access_mode::read_write>();
    auto __keys_rng = __keys_keep(__keys_first, __keys_last).all_view();
    auto __vals_keep = oneapi::dpl::__ranges::__get_sycl_range<sycl::access_mode::read_write>();
    auto __vals_rng = __vals_keep(__vals_first, __vals_first + __n).all_view();
    auto __pack = __impl::__rng_pack{std::move(__keys_rng), std::move(__vals_rng)};

    auto __keys_out_keep = oneapi::dpl::__ranges::__get_sycl_range<sycl::access_mode::read_write>();
    auto __keys_out_rng = __keys_out_keep(__keys_out_first, __keys_out_first + __n).all_view();
    auto __vals_out_keep = oneapi::dpl::__ranges::__get_sycl_range<sycl::access_mode::read_write>();
    auto __vals_out_rng = __vals_out_keep(__vals_out_first, __vals_out_first + __n).all_view();
    auto __pack_out = __impl::__rng_pack{std::move(__keys_out_rng), std::move(__vals_out_rng)};
    return __impl::__batched_merge_sort<__is_ascending>(__q, std::move(__pack), std::move(__pack_out), __segment_size,
                                                        __param);
}

} // namespace oneapi::dpl::experimental::kt::gpu

#endif // _ONEDPL_KT_BATCHED_MERGE_SORT_H
