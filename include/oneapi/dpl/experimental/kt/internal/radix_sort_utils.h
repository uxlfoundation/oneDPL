// -*- C++ -*-
//===-- sycl_radix_sort_utils.h ------------------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _ONEDPL_KT_SYCL_RADIX_SORT_UTILS_H
#define _ONEDPL_KT_SYCL_RADIX_SORT_UTILS_H

#include <limits>
#include <cstdint>
#include <type_traits>
#include <bit>

#include "../../../pstl/hetero/dpcpp/sycl_defs.h"
#include "oneapi/dpl/pstl/onedpl_config.h"
#include "../../../pstl/hetero/dpcpp/utils_ranges_sycl.h"

namespace oneapi::dpl::experimental::kt::gpu::__impl
{

struct __esimd_tag{};
struct __sycl_tag{};


//-----------------------------------------------------------------------------
// Parameter validation
//-----------------------------------------------------------------------------
template <std::uint8_t __radix_bits, std::uint16_t __data_per_workitem, std::uint16_t __workgroup_size>
constexpr void
__check_onesweep_params()
{
    static_assert(__radix_bits == 8);
    //static_assert(__data_per_workitem % 32 == 0);
    //static_assert(__workgroup_size == 32 || __workgroup_size == 64);
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
template <bool __is_ascending>
bool
__order_preserving_cast_scalar(bool __src)
{
    if constexpr (__is_ascending)
        return __src;
    else
        return !__src;
}

// Order-preserving cast for unsigned integers - scalar version
template <bool __is_ascending, typename _UInt, std::enable_if_t<std::is_unsigned_v<_UInt>, int> = 0>
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

// Order-preserving cast for 32-bit floats - scalar version
template <bool __is_ascending, typename _Float,
          std::enable_if_t<std::is_floating_point_v<_Float> && sizeof(_Float) == sizeof(std::uint32_t), int> = 0>
std::uint32_t
__order_preserving_cast_scalar(_Float __src)
{
    std::uint32_t __uint32_src = sycl::bit_cast<std::uint32_t>(__src);
    std::uint32_t __mask;
    bool __sign_bit_is_zero = (__uint32_src >> 31 == 0);
    if constexpr (__is_ascending)
    {
        __mask = __sign_bit_is_zero ? 0x80000000u : 0xFFFFFFFFu;
    }
    else
    {
        __mask = __sign_bit_is_zero ? 0x7FFFFFFFu : std::uint32_t(0);
    }
    return __uint32_src ^ __mask;
}

// Order-preserving cast for 64-bit floats - scalar version
template <bool __is_ascending, typename _Float,
          std::enable_if_t<std::is_floating_point_v<_Float> && sizeof(_Float) == sizeof(std::uint64_t), int> = 0>
std::uint64_t
__order_preserving_cast_scalar(_Float __src)
{
    std::uint64_t __uint64_src = sycl::bit_cast<std::uint64_t>(__src);
    std::uint64_t __mask;
    bool __sign_bit_is_zero = (__uint64_src >> 63 == 0);
    if constexpr (__is_ascending)
    {
        __mask = __sign_bit_is_zero ? 0x8000000000000000u : 0xFFFFFFFFFFFFFFFFu;
    }
    else
    {
        __mask = __sign_bit_is_zero ? 0x7FFFFFFFFFFFFFFFu : std::uint64_t(0);
    }
    return __uint64_src ^ __mask;
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
    if constexpr (::std::is_void_v<_T2>)
    {
        return __keys_pack<_N, _T1>{};
    }
    else
    {
        return __pairs_pack<_N, _T1, _T2>{};
    }
}

template <std::uint32_t __segment_width, std::uint32_t __num_segments, std::uint32_t __sub_group_size,
          typename _ScanBuffer>
void
__sub_group_cross_segment_exclusive_scan(sycl::sub_group& __sub_group, _ScanBuffer* __scan_elements)
{
    // TODO: make it work if this static assert is not true
    static_assert(__segment_width == __sub_group_size);
    using _ElemT = std::remove_reference_t<decltype(__scan_elements[0])>;
    _ElemT __carry = 0;
    auto __sub_group_local_id = __sub_group.get_local_linear_id();

    _ONEDPL_PRAGMA_UNROLL
    for (std::uint32_t __i = 0; __i < __num_segments; ++__i)
    {
        auto __element = __scan_elements[__i * __segment_width + __sub_group_local_id];
        auto __element_right_shift = sycl::shift_group_right(__sub_group, __element, 1);
        if (__sub_group_local_id == 0)
            __element_right_shift = 0;
        __scan_elements[__i * __segment_width + __sub_group_local_id] = __element_right_shift + __carry;

        __carry += sycl::group_broadcast(__sub_group, __element, __sub_group_size - 1);
    }
}

} // namespace oneapi::dpl::experimental::kt::gpu::__impl

#endif // _ONEDPL_KT_SYCL_RADIX_SORT_UTILS_H
