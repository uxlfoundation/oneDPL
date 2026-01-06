// -*- C++ -*-
//===-- sycl_radix_sort.h ------------------------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _ONEDPL_KT_SYCL_RADIX_SORT_H
#define _ONEDPL_KT_SYCL_RADIX_SORT_H

#include <cstdint>
#include <type_traits>

#include "../../pstl/hetero/dpcpp/utils_ranges_sycl.h"
#include "internal/sycl_radix_sort_utils.h"
#include "internal/sycl_radix_sort_dispatchers.h"
#include "../../pstl/utils.h"

namespace oneapi::dpl::experimental::kt::gpu::__sycl
{

// TODO: Add compile-time parameter validation
// TODO: Provide good default kernel parameters

//-----------------------------------------------------------------------------
// In-place sort (range-based)
//-----------------------------------------------------------------------------
template <bool __is_ascending = true, std::uint8_t __radix_bits = 8,
          typename _KernelParam, typename _KeysRng>
std::enable_if_t<!oneapi::dpl::__internal::__is_type_with_iterator_traits_v<_KeysRng>, sycl::event>
onesweep_sort(sycl::queue __q, _KeysRng&& __keys_rng, _KernelParam __param = {})
{
    if (__keys_rng.size() < 2)
        return {};

    auto __pack = __impl::__rng_pack{oneapi::dpl::__ranges::views::all(std::forward<_KeysRng>(__keys_rng))};
    return __impl::__radix_sort<__is_ascending, __radix_bits, /*__in_place=*/true>(
        __q, __pack, __pack, __param);
}

//-----------------------------------------------------------------------------
// In-place sort (iterator-based)
//-----------------------------------------------------------------------------
template <bool __is_ascending = true, std::uint8_t __radix_bits = 8,
          typename _KernelParam, typename _KeysIterator>
std::enable_if_t<oneapi::dpl::__internal::__is_type_with_iterator_traits_v<_KeysIterator>, sycl::event>
onesweep_sort(sycl::queue __q, _KeysIterator __keys_first, _KeysIterator __keys_last,
              _KernelParam __param = {})
{
    if (__keys_last - __keys_first < 2)
        return {};

    // Use read_write for in-place sorting
    auto __keys_keep = oneapi::dpl::__ranges::__get_sycl_range<
        sycl::access_mode::read_write, _KeysIterator>();
    auto __keys_rng = __keys_keep(__keys_first, __keys_last).all_view();
    auto __pack = __impl::__rng_pack{std::move(__keys_rng)};

    return __impl::__radix_sort<__is_ascending, __radix_bits, /*__in_place=*/true>(
        __q, __pack, __pack, __param);
}

//-----------------------------------------------------------------------------
// Out-of-place sort (range-based)
//-----------------------------------------------------------------------------
template <bool __is_ascending = true, std::uint8_t __radix_bits = 8,
          typename _KernelParam, typename _KeysRng1, typename _KeysRng2>
std::enable_if_t<!oneapi::dpl::__internal::__is_type_with_iterator_traits_v<_KeysRng1>, sycl::event>
onesweep_sort(sycl::queue __q, _KeysRng1&& __keys_rng, _KeysRng2&& __keys_rng_out,
              _KernelParam __param = {})
{
    if (__keys_rng.size() < 2)
        return {};

    auto __pack = __impl::__rng_pack{oneapi::dpl::__ranges::views::all(std::forward<_KeysRng1>(__keys_rng))};
    auto __pack_out = __impl::__rng_pack{oneapi::dpl::__ranges::views::all(std::forward<_KeysRng2>(__keys_rng_out))};

    return __impl::__radix_sort<__is_ascending, __radix_bits, /*__in_place=*/false>(
        __q, std::move(__pack), std::move(__pack_out), __param);
}

//-----------------------------------------------------------------------------
// Out-of-place sort (iterator-based)
//-----------------------------------------------------------------------------
template <bool __is_ascending = true, std::uint8_t __radix_bits = 8,
          typename _KernelParam,
          typename _KeysIterator1, typename _KeysIterator2>
std::enable_if_t<oneapi::dpl::__internal::__is_type_with_iterator_traits_v<_KeysIterator1>, sycl::event>
onesweep_sort(sycl::queue __q, _KeysIterator1 __keys_first, _KeysIterator1 __keys_last,
              _KeysIterator2 __keys_out_first, _KernelParam __param = {})
{
    auto __n = __keys_last - __keys_first;
    if (__n < 2)
        return {};

    // Use optimized access modes
    // Input: read-only
    auto __keys_keep = oneapi::dpl::__ranges::__get_sycl_range<
        sycl::access_mode::read, _KeysIterator1>();
    auto __keys_rng = __keys_keep(__keys_first, __keys_last).all_view();

    // Output: write (TODO: use no_init when PR #2551 is merged)
    auto __keys_out_keep = oneapi::dpl::__ranges::__get_sycl_range<
        sycl::access_mode::write, _KeysIterator2>();
    auto __keys_out_rng = __keys_out_keep(__keys_out_first, __keys_out_first + __n).all_view();

    auto __pack = __impl::__rng_pack{std::move(__keys_rng)};
    auto __pack_out = __impl::__rng_pack{std::move(__keys_out_rng)};

    return __impl::__radix_sort<__is_ascending, __radix_bits, /*__in_place=*/false>(
        __q, std::move(__pack), std::move(__pack_out), __param);
}

} // namespace oneapi::dpl::experimental::kt::gpu::__sycl

#endif // _ONEDPL_KT_SYCL_RADIX_SORT_H
