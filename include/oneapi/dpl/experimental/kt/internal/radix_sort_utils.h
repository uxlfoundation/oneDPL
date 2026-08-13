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

// Order-preserving cast for bool - scalar version
// Do not use bool directly - other unsupported types may be implicitly converted to bool
template <bool __is_ascending, typename _BoolT, std::enable_if_t<std::is_same_v<_BoolT, bool>, int> = 0>
bool
__order_preserving_cast_scalar(_BoolT __src)
{
    if constexpr (__is_ascending)
        return __src;
    else
        return !__src;
}

// Order-preserving cast for unsigned integers - scalar version
template <bool __is_ascending, typename _UInt,
          std::enable_if_t<std::is_unsigned_v<_UInt> && !std::is_same_v<_UInt, bool>, int> = 0>
_UInt
__order_preserving_cast_scalar(_UInt __src)
{
    if constexpr (__is_ascending)
        return __src;
    else
        return ~__src; // bitwise inversion
}

// Order-preserving cast for signed integers - scalar version
template <bool __is_ascending, typename _Int,
          std::enable_if_t<std::is_integral_v<_Int> && std::is_signed_v<_Int>, int> = 0>
std::make_unsigned_t<_Int>
__order_preserving_cast_scalar(_Int __src)
{
    using _UInt = std::make_unsigned_t<_Int>;
    // __mask: 100..0 for ascending, 011..1 for descending
    constexpr _UInt __mask =
        (__is_ascending) ? _UInt(1) << std::numeric_limits<_Int>::digits : std::numeric_limits<_UInt>::max() >> 1;
    return sycl::bit_cast<_UInt>(__src) ^ __mask;
}

template <std::size_t __size>
struct __uint_for_size;
template <> struct __uint_for_size<2> { using type = std::uint16_t; };
template <> struct __uint_for_size<4> { using type = std::uint32_t; };
template <> struct __uint_for_size<8> { using type = std::uint64_t; };
template <std::size_t __size>
using __uint_for_size_t = typename __uint_for_size<__size>::type;

template <typename _T>
inline constexpr bool __is_radix_sort_float_v =
    std::is_same_v<_T, sycl::half>
#if defined(SYCL_EXT_ONEAPI_BFLOAT16)
    || std::is_same_v<_T, sycl::ext::oneapi::bfloat16>
#endif // defined(SYCL_EXT_ONEAPI_BFLOAT16)
    || (std::is_floating_point_v<_T> && (sizeof(_T) == sizeof(std::uint32_t) || sizeof(_T) == sizeof(std::uint64_t)));

// Order-preserving cast for floating-point types - scalar version
template <bool __is_ascending, typename _Float, std::enable_if_t<__is_radix_sort_float_v<_Float>, int> = 0>
__uint_for_size_t<sizeof(_Float)>
__order_preserving_cast_scalar(_Float __src)
{
    using _UInt = __uint_for_size_t<sizeof(_Float)>;
    constexpr int __bits = std::numeric_limits<_UInt>::digits;
    constexpr _UInt __sign_mask = _UInt(1) << (__bits - 1);
    constexpr _UInt __magnitude_mask = _UInt(__sign_mask - 1);

    _UInt __uint_src = sycl::bit_cast<_UInt>(__src);
    // Map +0/-0 to the uppermost bit to place zero at the negative/positive boundary in its unsigned representation.
    if ((__uint_src & __magnitude_mask) == 0)
        return __sign_mask;
    _UInt __mask;
    if constexpr (__is_ascending)
        __mask = ((__uint_src & __sign_mask) == 0) ? __sign_mask : std::numeric_limits<_UInt>::max();
    else
        __mask = ((__uint_src & __sign_mask) == 0) ? __magnitude_mask : _UInt(0);
    return __uint_src ^ __mask;
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

// std::numeric_limits<_T>::max and std::numeric_limits<_T>::lowest cannot be used as an identity for
// performing radix sort of floating point numbers.
// They do not set the smallest exponent bit (i.e. the max is 7F7FFFFF for 32bit float),
// thus such an identity is not guaranteed to be put at the end of the sorted sequence after each radix sort stage,
// e.g. 00FF0000 numbers will be pushed out by 7F7FFFFF identities when sorting 16-23 bits.
template <typename _T, bool __is_ascending, std::enable_if_t<__is_radix_sort_float_v<_T>, int> = 0>
constexpr _T
__sort_identity()
{
    using _UInt = __uint_for_size_t<sizeof(_T)>;
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
