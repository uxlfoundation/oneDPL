// -*- C++ -*-
//===-- batched_stable_sort.h ---------------------------------------------===//
//
// Copyright (C) UXL Foundation Contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _ONEDPL_KT_BATCHED_STABLE_SORT_H
#define _ONEDPL_KT_BATCHED_STABLE_SORT_H

#include <cassert>
#include <cstdint>
#include <cstddef>
#include <iterator>
#include <type_traits>
#include <utility>

#include "../../pstl/hetero/dpcpp/utils_ranges_sycl.h"
#include "kernel_param.h"
#include "internal/kt_defs.h"
#include "internal/kt_utils.h"
#include "internal/radix_sort_utils.h"
#include "internal/radix_sort_dispatchers.h"

namespace oneapi::dpl::experimental::kt::gpu
{

//-----------------------------------------------------------------------------
// Sort configurations: each selects one algorithm and carries its specific parameters
//-----------------------------------------------------------------------------

// Modified OneSweep radix sort; any __segment_size
template <std::uint8_t __radix_bits = 8>
struct radix_many_groups_tag
{
    static constexpr std::uint8_t radix_bits = __radix_bits;
};

namespace __impl
{

template <typename _SortConfig>
struct __is_radix_many_groups_tag : std::false_type
{
};
template <std::uint8_t __radix_bits>
struct __is_radix_many_groups_tag<radix_many_groups_tag<__radix_bits>> : std::true_type
{
};

// Rejects unsupported combinations of _SortConfig and kernel_param at compile time
template <typename _SortConfig, typename _KernelParam>
constexpr void
__check_batched_sort_config()
{
    static_assert(__is_radix_many_groups_tag<_SortConfig>::value,
                  "Unsupported _SortConfig: only radix_many_groups_tag is currently implemented");
    if constexpr (__is_radix_many_groups_tag<_SortConfig>::value)
    {
        static_assert(_SortConfig::radix_bits == 8, "radix_many_groups_tag currently requires 8 radix bits");
        static_assert(_KernelParam::workgroup_size == 512 || _KernelParam::workgroup_size == 1024,
                      "radix_many_groups_tag currently requires 512 or 1024 work-items in a work-group");
    }
}

// Segment size type of the range overloads: the difference type of the keys input range
template <typename _Rng>
using __rng_difference_t = oneapi::dpl::__internal::__difference_t<std::decay_t<_Rng>>;

template <typename _SortConfig, typename _KeyT, typename _ValT, typename _KernelParam>
constexpr std::size_t
__batched_stable_sort_bytes_req(std::size_t __segment_size, std::size_t __total_size)
{
    __check_batched_sort_config<_SortConfig, _KernelParam>();
    if (__total_size == 0)
        return 0;
    assert(__segment_size > 0 && "The segment size must be greater than zero");
    return __onesweep_temp_bytes<_KeyT, _ValT, _SortConfig::radix_bits, _KernelParam::data_per_workitem,
                                 _KernelParam::workgroup_size>(__total_size, __segment_size);
}

template <typename _SortConfig, bool __is_ascending, bool __in_place, typename _RngPack1, typename _RngPack2,
          typename _KernelParam, typename _TempStorage>
sycl::event
__batched_stable_sort(sycl::queue __q, _RngPack1&& __pack_in, _RngPack2&& __pack_out, std::size_t __segment_size,
                      _KernelParam __param, _TempStorage __temp_storage)
{
    return __batched_onesweep<__is_ascending, _SortConfig::radix_bits, __in_place>(
        __q, std::forward<_RngPack1>(__pack_in), std::forward<_RngPack2>(__pack_out), __segment_size, __param,
        __temp_storage);
}

// The bodies of the public overloads below. _TempStorage is __internal_temp_storage or __external_temp_storage.

template <typename _SortConfig, bool __is_ascending, typename _KernelParam, typename _TempStorage, typename _KeysRng,
          typename _Size>
sycl::event
__batched_stable_sort_rng(sycl::queue __q, _KeysRng&& __keys_rng, _Size __segment_size, _KernelParam __param,
                          _TempStorage __temp_storage)
{
    __check_batched_sort_config<_SortConfig, _KernelParam>();
#if _ONEDPL_ENABLE_SYCL_RADIX_SORT_KT
    auto __n = __keys_rng.size();
    __check_batched_sort_params(__n, __segment_size);
    if (__n == 0)
        return {};

    auto __pack = __rng_pack{oneapi::dpl::__ranges::views::all(std::forward<_KeysRng>(__keys_rng))};
    return __batched_stable_sort<_SortConfig, __is_ascending, /*__in_place=*/true>(__q, __pack, __pack, __segment_size,
                                                                                   __param, __temp_storage);
#else
    __sycl_radix_sort_unsupported_msg<_KeysRng>();
#endif
}

template <typename _SortConfig, bool __is_ascending, typename _KernelParam, typename _TempStorage,
          typename _KeysIterator, typename _Size>
sycl::event
__batched_stable_sort_iter(sycl::queue __q, _KeysIterator __keys_first, _KeysIterator __keys_last, _Size __segment_size,
                           _KernelParam __param, _TempStorage __temp_storage)
{
    __check_batched_sort_config<_SortConfig, _KernelParam>();
#if _ONEDPL_ENABLE_SYCL_RADIX_SORT_KT
    auto __n = __keys_last - __keys_first;
    __check_batched_sort_params(__n, __segment_size);
    if (__n == 0)
        return {};

    auto __keys_keep = oneapi::dpl::__ranges::__get_sycl_range<sycl::access_mode::read_write>();
    auto __keys_rng = __keys_keep(__keys_first, __keys_last).all_view();
    auto __pack = __rng_pack{std::move(__keys_rng)};
    return __batched_stable_sort<_SortConfig, __is_ascending, /*__in_place=*/true>(__q, __pack, __pack, __segment_size,
                                                                                   __param, __temp_storage);
#else
    __sycl_radix_sort_unsupported_msg<_KeysIterator>();
#endif
}

template <typename _SortConfig, bool __is_ascending, typename _KernelParam, typename _TempStorage, typename _KeysRng,
          typename _ValsRng, typename _Size>
sycl::event
__batched_stable_sort_by_key_rng(sycl::queue __q, _KeysRng&& __keys_rng, _ValsRng&& __vals_rng, _Size __segment_size,
                                 _KernelParam __param, _TempStorage __temp_storage)
{
    __check_batched_sort_config<_SortConfig, _KernelParam>();
#if _ONEDPL_ENABLE_SYCL_RADIX_SORT_KT
    auto __n = __keys_rng.size();
    __check_batched_sort_params(__n, __segment_size);
    if (__n == 0)
        return {};

    auto __pack = __rng_pack{oneapi::dpl::__ranges::views::all(std::forward<_KeysRng>(__keys_rng)),
                             oneapi::dpl::__ranges::views::all(std::forward<_ValsRng>(__vals_rng))};
    return __batched_stable_sort<_SortConfig, __is_ascending, /*__in_place=*/true>(__q, __pack, __pack, __segment_size,
                                                                                   __param, __temp_storage);
#else
    __sycl_radix_sort_unsupported_msg<_KeysRng>();
#endif
}

template <typename _SortConfig, bool __is_ascending, typename _KernelParam, typename _TempStorage,
          typename _KeysIterator, typename _ValsIterator, typename _Size>
sycl::event
__batched_stable_sort_by_key_iter(sycl::queue __q, _KeysIterator __keys_first, _KeysIterator __keys_last,
                                  _ValsIterator __vals_first, _Size __segment_size, _KernelParam __param,
                                  _TempStorage __temp_storage)
{
    __check_batched_sort_config<_SortConfig, _KernelParam>();
#if _ONEDPL_ENABLE_SYCL_RADIX_SORT_KT
    auto __n = __keys_last - __keys_first;
    __check_batched_sort_params(__n, __segment_size);
    if (__n == 0)
        return {};

    auto __keys_keep = oneapi::dpl::__ranges::__get_sycl_range<sycl::access_mode::read_write>();
    auto __keys_rng = __keys_keep(__keys_first, __keys_last).all_view();
    auto __vals_keep = oneapi::dpl::__ranges::__get_sycl_range<sycl::access_mode::read_write>();
    auto __vals_rng = __vals_keep(__vals_first, __vals_first + __n).all_view();
    auto __pack = __rng_pack{std::move(__keys_rng), std::move(__vals_rng)};
    return __batched_stable_sort<_SortConfig, __is_ascending, /*__in_place=*/true>(__q, __pack, __pack, __segment_size,
                                                                                   __param, __temp_storage);
#else
    __sycl_radix_sort_unsupported_msg<_KeysIterator>();
#endif
}

template <typename _SortConfig, bool __is_ascending, typename _KernelParam, typename _TempStorage, typename _KeysRng1,
          typename _KeysRng2, typename _Size>
sycl::event
__batched_stable_sort_out_rng(sycl::queue __q, _KeysRng1&& __keys_rng, _KeysRng2&& __keys_out_rng, _Size __segment_size,
                              _KernelParam __param, _TempStorage __temp_storage)
{
    __check_batched_sort_config<_SortConfig, _KernelParam>();
#if _ONEDPL_ENABLE_SYCL_RADIX_SORT_KT
    auto __n = __keys_rng.size();
    __check_batched_sort_params(__n, __segment_size);
    if (__n == 0)
        return {};

    auto __pack = __rng_pack{oneapi::dpl::__ranges::views::all(std::forward<_KeysRng1>(__keys_rng))};
    auto __pack_out = __rng_pack{oneapi::dpl::__ranges::views::all(std::forward<_KeysRng2>(__keys_out_rng))};
    return __batched_stable_sort<_SortConfig, __is_ascending, /*__in_place=*/false>(
        __q, std::move(__pack), std::move(__pack_out), __segment_size, __param, __temp_storage);
#else
    __sycl_radix_sort_unsupported_msg<_KeysRng1>();
#endif
}

template <typename _SortConfig, bool __is_ascending, typename _KernelParam, typename _TempStorage,
          typename _KeysIterator1, typename _KeysIterator2, typename _Size>
sycl::event
__batched_stable_sort_out_iter(sycl::queue __q, _KeysIterator1 __keys_first, _KeysIterator1 __keys_last,
                               _KeysIterator2 __keys_out_first, _Size __segment_size, _KernelParam __param,
                               _TempStorage __temp_storage)
{
    __check_batched_sort_config<_SortConfig, _KernelParam>();
#if _ONEDPL_ENABLE_SYCL_RADIX_SORT_KT
    auto __n = __keys_last - __keys_first;
    __check_batched_sort_params(__n, __segment_size);
    if (__n == 0)
        return {};

    auto __keys_keep = oneapi::dpl::__ranges::__get_sycl_range<sycl::access_mode::read_write>();
    auto __keys_rng = __keys_keep(__keys_first, __keys_last).all_view();
    auto __pack = __rng_pack{std::move(__keys_rng)};
    auto __keys_out_keep = oneapi::dpl::__ranges::__get_sycl_range<sycl::access_mode::read_write>();
    auto __keys_out_rng = __keys_out_keep(__keys_out_first, __keys_out_first + __n).all_view();
    auto __pack_out = __rng_pack{std::move(__keys_out_rng)};
    return __batched_stable_sort<_SortConfig, __is_ascending, /*__in_place=*/false>(
        __q, std::move(__pack), std::move(__pack_out), __segment_size, __param, __temp_storage);
#else
    __sycl_radix_sort_unsupported_msg<_KeysIterator1>();
#endif
}

template <typename _SortConfig, bool __is_ascending, typename _KernelParam, typename _TempStorage, typename _KeysRng1,
          typename _ValsRng1, typename _KeysRng2, typename _ValsRng2, typename _Size>
sycl::event
__batched_stable_sort_by_key_out_rng(sycl::queue __q, _KeysRng1&& __keys_rng, _ValsRng1&& __vals_rng,
                                     _KeysRng2&& __keys_out_rng, _ValsRng2&& __vals_out_rng, _Size __segment_size,
                                     _KernelParam __param, _TempStorage __temp_storage)
{
    __check_batched_sort_config<_SortConfig, _KernelParam>();
#if _ONEDPL_ENABLE_SYCL_RADIX_SORT_KT
    auto __n = __keys_rng.size();
    __check_batched_sort_params(__n, __segment_size);
    if (__n == 0)
        return {};

    auto __pack = __rng_pack{oneapi::dpl::__ranges::views::all(std::forward<_KeysRng1>(__keys_rng)),
                             oneapi::dpl::__ranges::views::all(std::forward<_ValsRng1>(__vals_rng))};
    auto __pack_out = __rng_pack{oneapi::dpl::__ranges::views::all(std::forward<_KeysRng2>(__keys_out_rng)),
                                 oneapi::dpl::__ranges::views::all(std::forward<_ValsRng2>(__vals_out_rng))};
    return __batched_stable_sort<_SortConfig, __is_ascending, /*__in_place=*/false>(
        __q, std::move(__pack), std::move(__pack_out), __segment_size, __param, __temp_storage);
#else
    __sycl_radix_sort_unsupported_msg<_KeysRng1>();
#endif
}

template <typename _SortConfig, bool __is_ascending, typename _KernelParam, typename _TempStorage,
          typename _KeysIterator1, typename _ValsIterator1, typename _KeysIterator2, typename _ValsIterator2,
          typename _Size>
sycl::event
__batched_stable_sort_by_key_out_iter(sycl::queue __q, _KeysIterator1 __keys_first, _KeysIterator1 __keys_last,
                                      _ValsIterator1 __vals_first, _KeysIterator2 __keys_out_first,
                                      _ValsIterator2 __vals_out_first, _Size __segment_size, _KernelParam __param,
                                      _TempStorage __temp_storage)
{
    __check_batched_sort_config<_SortConfig, _KernelParam>();
#if _ONEDPL_ENABLE_SYCL_RADIX_SORT_KT
    auto __n = __keys_last - __keys_first;
    __check_batched_sort_params(__n, __segment_size);
    if (__n == 0)
        return {};

    auto __keys_keep = oneapi::dpl::__ranges::__get_sycl_range<sycl::access_mode::read_write>();
    auto __keys_rng = __keys_keep(__keys_first, __keys_last).all_view();
    auto __vals_keep = oneapi::dpl::__ranges::__get_sycl_range<sycl::access_mode::read_write>();
    auto __vals_rng = __vals_keep(__vals_first, __vals_first + __n).all_view();
    auto __pack = __rng_pack{std::move(__keys_rng), std::move(__vals_rng)};

    auto __keys_out_keep = oneapi::dpl::__ranges::__get_sycl_range<sycl::access_mode::read_write>();
    auto __keys_out_rng = __keys_out_keep(__keys_out_first, __keys_out_first + __n).all_view();
    auto __vals_out_keep = oneapi::dpl::__ranges::__get_sycl_range<sycl::access_mode::read_write>();
    auto __vals_out_rng = __vals_out_keep(__vals_out_first, __vals_out_first + __n).all_view();
    auto __pack_out = __rng_pack{std::move(__keys_out_rng), std::move(__vals_out_rng)};
    return __batched_stable_sort<_SortConfig, __is_ascending, /*__in_place=*/false>(
        __q, std::move(__pack), std::move(__pack_out), __segment_size, __param, __temp_storage);
#else
    __sycl_radix_sort_unsupported_msg<_KeysIterator1>();
#endif
}

} // namespace __impl

//-----------------------------------------------------------------------------
// Temporary memory requirements
//-----------------------------------------------------------------------------
// Bytes of temporary global memory needed to sort __total_size elements in segments of __segment_size elements, to be
// provided to the overloads which accept a temporary storage pointer and its size in bytes (directly before the
// kernel_param). The storage is USM device or shared memory with no alignment requirement. The sort does not allocate
// or free it, and it must stay valid and must not be used otherwise until the returned event completes.

template <typename _SortConfig, typename _KeyT, typename _KernelParam>
constexpr std::size_t
batched_stable_sort_bytes_req(std::size_t __segment_size, std::size_t __total_size, [[maybe_unused]] bool __in_place,
                              _KernelParam = {})
{
    return __impl::__batched_stable_sort_bytes_req<_SortConfig, _KeyT, void, _KernelParam>(__segment_size,
                                                                                           __total_size);
}

template <typename _SortConfig, typename _KeyT, typename _ValT, typename _KernelParam>
constexpr std::size_t
batched_stable_sort_by_key_bytes_req(std::size_t __segment_size, std::size_t __total_size,
                                     [[maybe_unused]] bool __in_place, _KernelParam = {})
{
    return __impl::__batched_stable_sort_bytes_req<_SortConfig, _KeyT, _ValT, _KernelParam>(__segment_size,
                                                                                            __total_size);
}

//-----------------------------------------------------------------------------
// In-place sorts
//-----------------------------------------------------------------------------

template <typename _SortConfig, bool __is_ascending = true, typename _KernelParam, typename _KeysRng>
std::enable_if_t<!oneapi::dpl::__internal::__is_type_with_iterator_traits_v<_KeysRng> &&
                     __impl::__is_kernel_param_v<_KernelParam>,
                 sycl::event>
batched_stable_sort(sycl::queue __q, _KeysRng&& __keys_rng, __impl::__rng_difference_t<_KeysRng> __segment_size,
                    _KernelParam __param = {})
{
    return __impl::__batched_stable_sort_rng<_SortConfig, __is_ascending>(
        __q, std::forward<_KeysRng>(__keys_rng), __segment_size, __param, __impl::__internal_temp_storage{});
}

template <typename _SortConfig, bool __is_ascending = true, typename _KernelParam, typename _KeysRng>
std::enable_if_t<!oneapi::dpl::__internal::__is_type_with_iterator_traits_v<_KeysRng> &&
                     __impl::__is_kernel_param_v<_KernelParam>,
                 sycl::event>
batched_stable_sort(sycl::queue __q, _KeysRng&& __keys_rng, __impl::__rng_difference_t<_KeysRng> __segment_size,
                    void* __temp_storage, std::size_t __temp_storage_bytes, _KernelParam __param = {})
{
    return __impl::__batched_stable_sort_rng<_SortConfig, __is_ascending>(
        __q, std::forward<_KeysRng>(__keys_rng), __segment_size, __param,
        __impl::__external_temp_storage{__temp_storage, __temp_storage_bytes});
}

template <typename _SortConfig, bool __is_ascending = true, typename _KernelParam, typename _KeysIterator>
std::enable_if_t<oneapi::dpl::__internal::__is_type_with_iterator_traits_v<_KeysIterator> &&
                     __impl::__is_kernel_param_v<_KernelParam>,
                 sycl::event>
batched_stable_sort(sycl::queue __q, _KeysIterator __keys_first, _KeysIterator __keys_last,
                    typename std::iterator_traits<_KeysIterator>::difference_type __segment_size,
                    _KernelParam __param = {})
{
    return __impl::__batched_stable_sort_iter<_SortConfig, __is_ascending>(
        __q, __keys_first, __keys_last, __segment_size, __param, __impl::__internal_temp_storage{});
}

template <typename _SortConfig, bool __is_ascending = true, typename _KernelParam, typename _KeysIterator>
std::enable_if_t<oneapi::dpl::__internal::__is_type_with_iterator_traits_v<_KeysIterator> &&
                     __impl::__is_kernel_param_v<_KernelParam>,
                 sycl::event>
batched_stable_sort(sycl::queue __q, _KeysIterator __keys_first, _KeysIterator __keys_last,
                    typename std::iterator_traits<_KeysIterator>::difference_type __segment_size, void* __temp_storage,
                    std::size_t __temp_storage_bytes, _KernelParam __param = {})
{
    return __impl::__batched_stable_sort_iter<_SortConfig, __is_ascending>(
        __q, __keys_first, __keys_last, __segment_size, __param,
        __impl::__external_temp_storage{__temp_storage, __temp_storage_bytes});
}

template <typename _SortConfig, bool __is_ascending = true, typename _KernelParam, typename _KeysRng, typename _ValsRng>
std::enable_if_t<!oneapi::dpl::__internal::__is_type_with_iterator_traits_v<_KeysRng> &&
                     __impl::__is_kernel_param_v<_KernelParam>,
                 sycl::event>
batched_stable_sort_by_key(sycl::queue __q, _KeysRng&& __keys_rng, _ValsRng&& __vals_rng,
                           __impl::__rng_difference_t<_KeysRng> __segment_size, _KernelParam __param = {})
{
    return __impl::__batched_stable_sort_by_key_rng<_SortConfig, __is_ascending>(
        __q, std::forward<_KeysRng>(__keys_rng), std::forward<_ValsRng>(__vals_rng), __segment_size, __param,
        __impl::__internal_temp_storage{});
}

template <typename _SortConfig, bool __is_ascending = true, typename _KernelParam, typename _KeysRng, typename _ValsRng>
std::enable_if_t<!oneapi::dpl::__internal::__is_type_with_iterator_traits_v<_KeysRng> &&
                     __impl::__is_kernel_param_v<_KernelParam>,
                 sycl::event>
batched_stable_sort_by_key(sycl::queue __q, _KeysRng&& __keys_rng, _ValsRng&& __vals_rng,
                           __impl::__rng_difference_t<_KeysRng> __segment_size, void* __temp_storage,
                           std::size_t __temp_storage_bytes, _KernelParam __param = {})
{
    return __impl::__batched_stable_sort_by_key_rng<_SortConfig, __is_ascending>(
        __q, std::forward<_KeysRng>(__keys_rng), std::forward<_ValsRng>(__vals_rng), __segment_size, __param,
        __impl::__external_temp_storage{__temp_storage, __temp_storage_bytes});
}

template <typename _SortConfig, bool __is_ascending = true, typename _KernelParam, typename _KeysIterator,
          typename _ValsIterator>
std::enable_if_t<oneapi::dpl::__internal::__is_type_with_iterator_traits_v<_KeysIterator> &&
                     __impl::__is_kernel_param_v<_KernelParam>,
                 sycl::event>
batched_stable_sort_by_key(sycl::queue __q, _KeysIterator __keys_first, _KeysIterator __keys_last,
                           _ValsIterator __vals_first,
                           typename std::iterator_traits<_KeysIterator>::difference_type __segment_size,
                           _KernelParam __param = {})
{
    return __impl::__batched_stable_sort_by_key_iter<_SortConfig, __is_ascending>(
        __q, __keys_first, __keys_last, __vals_first, __segment_size, __param, __impl::__internal_temp_storage{});
}

template <typename _SortConfig, bool __is_ascending = true, typename _KernelParam, typename _KeysIterator,
          typename _ValsIterator>
std::enable_if_t<oneapi::dpl::__internal::__is_type_with_iterator_traits_v<_KeysIterator> &&
                     __impl::__is_kernel_param_v<_KernelParam>,
                 sycl::event>
batched_stable_sort_by_key(sycl::queue __q, _KeysIterator __keys_first, _KeysIterator __keys_last,
                           _ValsIterator __vals_first,
                           typename std::iterator_traits<_KeysIterator>::difference_type __segment_size,
                           void* __temp_storage, std::size_t __temp_storage_bytes, _KernelParam __param = {})
{
    return __impl::__batched_stable_sort_by_key_iter<_SortConfig, __is_ascending>(
        __q, __keys_first, __keys_last, __vals_first, __segment_size, __param,
        __impl::__external_temp_storage{__temp_storage, __temp_storage_bytes});
}

//-----------------------------------------------------------------------------
// Out-of-place sorts
//-----------------------------------------------------------------------------
// The out-of-place key-value overloads require an output keys sequence which is not an integer, as otherwise an
// in-place call with temporary storage (keys, values, segment size, pointer, bytes) would also match them.

template <typename _SortConfig, bool __is_ascending = true, typename _KernelParam, typename _KeysRng1,
          typename _KeysRng2>
std::enable_if_t<!oneapi::dpl::__internal::__is_type_with_iterator_traits_v<_KeysRng1> &&
                     __impl::__is_kernel_param_v<_KernelParam>,
                 sycl::event>
batched_stable_sort(sycl::queue __q, _KeysRng1&& __keys_rng, _KeysRng2&& __keys_out_rng,
                    __impl::__rng_difference_t<_KeysRng1> __segment_size, _KernelParam __param = {})
{
    return __impl::__batched_stable_sort_out_rng<_SortConfig, __is_ascending>(
        __q, std::forward<_KeysRng1>(__keys_rng), std::forward<_KeysRng2>(__keys_out_rng), __segment_size, __param,
        __impl::__internal_temp_storage{});
}

template <typename _SortConfig, bool __is_ascending = true, typename _KernelParam, typename _KeysRng1,
          typename _KeysRng2>
std::enable_if_t<!oneapi::dpl::__internal::__is_type_with_iterator_traits_v<_KeysRng1> &&
                     __impl::__is_kernel_param_v<_KernelParam>,
                 sycl::event>
batched_stable_sort(sycl::queue __q, _KeysRng1&& __keys_rng, _KeysRng2&& __keys_out_rng,
                    __impl::__rng_difference_t<_KeysRng1> __segment_size, void* __temp_storage,
                    std::size_t __temp_storage_bytes, _KernelParam __param = {})
{
    return __impl::__batched_stable_sort_out_rng<_SortConfig, __is_ascending>(
        __q, std::forward<_KeysRng1>(__keys_rng), std::forward<_KeysRng2>(__keys_out_rng), __segment_size, __param,
        __impl::__external_temp_storage{__temp_storage, __temp_storage_bytes});
}

template <typename _SortConfig, bool __is_ascending = true, typename _KernelParam, typename _KeysIterator1,
          typename _KeysIterator2>
std::enable_if_t<oneapi::dpl::__internal::__is_type_with_iterator_traits_v<_KeysIterator1> &&
                     __impl::__is_kernel_param_v<_KernelParam>,
                 sycl::event>
batched_stable_sort(sycl::queue __q, _KeysIterator1 __keys_first, _KeysIterator1 __keys_last,
                    _KeysIterator2 __keys_out_first,
                    typename std::iterator_traits<_KeysIterator1>::difference_type __segment_size,
                    _KernelParam __param = {})
{
    return __impl::__batched_stable_sort_out_iter<_SortConfig, __is_ascending>(
        __q, __keys_first, __keys_last, __keys_out_first, __segment_size, __param, __impl::__internal_temp_storage{});
}

template <typename _SortConfig, bool __is_ascending = true, typename _KernelParam, typename _KeysIterator1,
          typename _KeysIterator2>
std::enable_if_t<oneapi::dpl::__internal::__is_type_with_iterator_traits_v<_KeysIterator1> &&
                     __impl::__is_kernel_param_v<_KernelParam>,
                 sycl::event>
batched_stable_sort(sycl::queue __q, _KeysIterator1 __keys_first, _KeysIterator1 __keys_last,
                    _KeysIterator2 __keys_out_first,
                    typename std::iterator_traits<_KeysIterator1>::difference_type __segment_size, void* __temp_storage,
                    std::size_t __temp_storage_bytes, _KernelParam __param = {})
{
    return __impl::__batched_stable_sort_out_iter<_SortConfig, __is_ascending>(
        __q, __keys_first, __keys_last, __keys_out_first, __segment_size, __param,
        __impl::__external_temp_storage{__temp_storage, __temp_storage_bytes});
}

template <typename _SortConfig, bool __is_ascending = true, typename _KernelParam, typename _KeysRng1,
          typename _ValsRng1, typename _KeysRng2, typename _ValsRng2>
std::enable_if_t<!oneapi::dpl::__internal::__is_type_with_iterator_traits_v<_KeysRng1> &&
                     !std::is_integral_v<std::decay_t<_KeysRng2>> && __impl::__is_kernel_param_v<_KernelParam>,
                 sycl::event>
batched_stable_sort_by_key(sycl::queue __q, _KeysRng1&& __keys_rng, _ValsRng1&& __vals_rng, _KeysRng2&& __keys_out_rng,
                           _ValsRng2&& __vals_out_rng, __impl::__rng_difference_t<_KeysRng1> __segment_size,
                           _KernelParam __param = {})
{
    return __impl::__batched_stable_sort_by_key_out_rng<_SortConfig, __is_ascending>(
        __q, std::forward<_KeysRng1>(__keys_rng), std::forward<_ValsRng1>(__vals_rng),
        std::forward<_KeysRng2>(__keys_out_rng), std::forward<_ValsRng2>(__vals_out_rng), __segment_size, __param,
        __impl::__internal_temp_storage{});
}

template <typename _SortConfig, bool __is_ascending = true, typename _KernelParam, typename _KeysRng1,
          typename _ValsRng1, typename _KeysRng2, typename _ValsRng2>
std::enable_if_t<!oneapi::dpl::__internal::__is_type_with_iterator_traits_v<_KeysRng1> &&
                     !std::is_integral_v<std::decay_t<_KeysRng2>> && __impl::__is_kernel_param_v<_KernelParam>,
                 sycl::event>
batched_stable_sort_by_key(sycl::queue __q, _KeysRng1&& __keys_rng, _ValsRng1&& __vals_rng, _KeysRng2&& __keys_out_rng,
                           _ValsRng2&& __vals_out_rng, __impl::__rng_difference_t<_KeysRng1> __segment_size,
                           void* __temp_storage, std::size_t __temp_storage_bytes, _KernelParam __param = {})
{
    return __impl::__batched_stable_sort_by_key_out_rng<_SortConfig, __is_ascending>(
        __q, std::forward<_KeysRng1>(__keys_rng), std::forward<_ValsRng1>(__vals_rng),
        std::forward<_KeysRng2>(__keys_out_rng), std::forward<_ValsRng2>(__vals_out_rng), __segment_size, __param,
        __impl::__external_temp_storage{__temp_storage, __temp_storage_bytes});
}

template <typename _SortConfig, bool __is_ascending = true, typename _KernelParam, typename _KeysIterator1,
          typename _ValsIterator1, typename _KeysIterator2, typename _ValsIterator2>
std::enable_if_t<oneapi::dpl::__internal::__is_type_with_iterator_traits_v<_KeysIterator1> &&
                     oneapi::dpl::__internal::__is_type_with_iterator_traits_v<_KeysIterator2> &&
                     __impl::__is_kernel_param_v<_KernelParam>,
                 sycl::event>
batched_stable_sort_by_key(sycl::queue __q, _KeysIterator1 __keys_first, _KeysIterator1 __keys_last,
                           _ValsIterator1 __vals_first, _KeysIterator2 __keys_out_first,
                           _ValsIterator2 __vals_out_first,
                           typename std::iterator_traits<_KeysIterator1>::difference_type __segment_size,
                           _KernelParam __param = {})
{
    return __impl::__batched_stable_sort_by_key_out_iter<_SortConfig, __is_ascending>(
        __q, __keys_first, __keys_last, __vals_first, __keys_out_first, __vals_out_first, __segment_size, __param,
        __impl::__internal_temp_storage{});
}

template <typename _SortConfig, bool __is_ascending = true, typename _KernelParam, typename _KeysIterator1,
          typename _ValsIterator1, typename _KeysIterator2, typename _ValsIterator2>
std::enable_if_t<oneapi::dpl::__internal::__is_type_with_iterator_traits_v<_KeysIterator1> &&
                     oneapi::dpl::__internal::__is_type_with_iterator_traits_v<_KeysIterator2> &&
                     __impl::__is_kernel_param_v<_KernelParam>,
                 sycl::event>
batched_stable_sort_by_key(sycl::queue __q, _KeysIterator1 __keys_first, _KeysIterator1 __keys_last,
                           _ValsIterator1 __vals_first, _KeysIterator2 __keys_out_first,
                           _ValsIterator2 __vals_out_first,
                           typename std::iterator_traits<_KeysIterator1>::difference_type __segment_size,
                           void* __temp_storage, std::size_t __temp_storage_bytes, _KernelParam __param = {})
{
    return __impl::__batched_stable_sort_by_key_out_iter<_SortConfig, __is_ascending>(
        __q, __keys_first, __keys_last, __vals_first, __keys_out_first, __vals_out_first, __segment_size, __param,
        __impl::__external_temp_storage{__temp_storage, __temp_storage_bytes});
}

} // namespace oneapi::dpl::experimental::kt::gpu

#endif // _ONEDPL_KT_BATCHED_STABLE_SORT_H
