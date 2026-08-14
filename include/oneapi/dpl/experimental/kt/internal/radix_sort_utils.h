// -*- C++ -*-
//===---------------------------------------------------------------------===//
//
// Copyright (C) UXL Foundation Contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _ONEDPL_KT_SYCL_RADIX_SORT_UTILS_H
#define _ONEDPL_KT_SYCL_RADIX_SORT_UTILS_H

#include <limits>
#include <cstdint>
#include <type_traits>
#include <cassert>

#include "../../../pstl/hetero/dpcpp/sycl_defs.h"
#include "oneapi/dpl/pstl/onedpl_config.h"
#include "../../../pstl/hetero/dpcpp/utils_ranges_sycl.h"
#include "../../../pstl/utils.h"
#include "kt_defs.h"

namespace oneapi::dpl::experimental::kt::gpu::__impl
{

namespace syclex = sycl::ext::oneapi::experimental;

struct __esimd_tag
{
};
struct __sycl_tag
{
};

//-----------------------------------------------------------------------------
// Tag-specific histogram kernel configuration
//-----------------------------------------------------------------------------
template <typename _KtTag>
struct __radix_sort_histogram_params;

template <>
struct __radix_sort_histogram_params<__esimd_tag>
{
    // Occupies all 64 XE cores on PVC-1550 tile
    static constexpr std::uint32_t __work_group_count = 64;
    // 64 XVEs ~ 2048 SIMD lanes. Each work group fully controls Xe core
    static constexpr std::uint32_t __work_group_size = 64;
};

template <>
struct __radix_sort_histogram_params<__sycl_tag>
{
    // Guarantees full hardware occupancy on PVC with oversubscription showing improved performance
    static constexpr std::uint32_t __work_group_count = 128 * 10;
    // Max work-group size in SYCL gives us control over 1024 lanes, allowing 2 work-groups per Xe core
    static constexpr std::uint32_t __work_group_size = 1024;
};

//-----------------------------------------------------------------------------
// Parameter validation
//-----------------------------------------------------------------------------
template <std::uint8_t __radix_bits, std::uint16_t __data_per_workitem, std::uint16_t __workgroup_size>
inline void
__check_sycl_sort_params([[maybe_unused]] std::size_t __n)
{
    static_assert(__radix_bits == 8);
    static_assert(__workgroup_size == 1024 || __workgroup_size == 512);
    assert((__n < (1 << 30)) && "Inputs >= 2^30 are currently unsupported in the SYCL sort KT");
}

template <typename _T>
constexpr void
__sycl_radix_sort_unsupported_msg()
{
    static_assert(oneapi::dpl::__internal::__always_false_v<_T>,
                  "oneDPL's SYCL radix sort kernel templates require SYCL_EXT_ONEAPI_SUB_GROUP_MASK, "
                  "SYCL_EXT_ONEAPI_FORWARD_PROGRESS, and SYCL_EXT_ONEAPI_ROOT_GROUP extension support. "
                  "Please use a oneAPI compiler version that supports these extensions. If using the Intel "
                  "oneAPI DPC++/C++ Compiler, a minimum version of 2025.1.0 is also required.");
}

//-----------------------------------------------------------------------------
// Scalar utility functions for pure SYCL kernels
//-----------------------------------------------------------------------------

// Get bits value (bucket) in a certain radix position - scalar version
template <std::uint16_t __radix_mask, typename _T, std::enable_if_t<std::is_unsigned_v<_T>, int> = 0>
std::uint16_t
__get_bucket_scalar(_T __value, std::uint32_t __radix_offset)
{
    return std::uint16_t(__value >> __radix_offset) & __radix_mask;
}

//-----------------------------------------------------------------------------
// Sort identity values - used to pad incomplete blocks during sorting
//-----------------------------------------------------------------------------
template <typename _T, bool __is_ascending, std::enable_if_t<std::is_integral_v<_T>, int> = 0>
constexpr _T
__sort_identity()
{
    if constexpr (__is_ascending)
        return std::numeric_limits<_T>::max();
    else
        return std::numeric_limits<_T>::lowest();
}

template <typename _T, bool __is_ascending,
          std::enable_if_t<oneapi::dpl::__internal::__is_radix_sort_float_v<_T>, int> = 0>
constexpr _T
__sort_identity()
{
    using _UInt = oneapi::dpl::__internal::__uint_for_size_t<sizeof(_T)>;
    if constexpr (__is_ascending)
        return sycl::bit_cast<_T>(_UInt(std::numeric_limits<_UInt>::max() >> 1));
    else
        return sycl::bit_cast<_T>(std::numeric_limits<_UInt>::max());
}

template <std::uint16_t _N, typename _KeyT>
struct __keys_pack
{
    _KeyT __keys[_N];
};

template <std::uint16_t _N, typename _KeyT, typename _ValT>
struct __pairs_pack
{
    _KeyT __keys[_N];
    _ValT __vals[_N];
};

template <std::uint16_t _N, typename _T1, typename _T2 = void>
auto
__make_key_value_pack()
{
    if constexpr (std::is_void_v<_T2>)
    {
        return __keys_pack<_N, _T1>{};
    }
    else
    {
        return __pairs_pack<_N, _T1, _T2>{};
    }
}

} // namespace oneapi::dpl::experimental::kt::gpu::__impl

#endif // _ONEDPL_KT_SYCL_RADIX_SORT_UTILS_H
