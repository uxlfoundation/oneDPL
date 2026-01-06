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

namespace oneapi::dpl::experimental::kt::gpu::__sycl::__impl
{

//-----------------------------------------------------------------------------
// Parameter validation
//-----------------------------------------------------------------------------
template <std::uint8_t __radix_bits, std::uint16_t __data_per_workitem, std::uint16_t __workgroup_size>
constexpr void
__check_onesweep_params()
{
    static_assert(__radix_bits == 8, "Only 8-bit radix is currently supported");
    static_assert(__data_per_workitem >= 32 && __data_per_workitem % 32 == 0,
                  "data_per_workitem must be >= 32 and divisible by 32");
    static_assert(__workgroup_size == 32 || __workgroup_size == 64 || __workgroup_size == 128,
                  "workgroup_size must be 32, 64, or 128");
}

//-----------------------------------------------------------------------------
// Sort identity values (for padding out-of-bounds elements)
//-----------------------------------------------------------------------------

// For integral types: use max/min depending on sort direction
template <typename _T, bool __is_ascending, std::enable_if_t<std::is_integral_v<_T>, int> = 0>
constexpr _T
__sort_identity()
{
    if constexpr (__is_ascending)
        return std::numeric_limits<_T>::max();
    else
        return std::numeric_limits<_T>::lowest();
}

// For floating point: use special bit patterns to avoid exponent issues
// (see ESIMD comments: max() doesn't set smallest exponent bit)
template <typename _T, bool __is_ascending,
          std::enable_if_t<std::is_floating_point_v<_T> && sizeof(_T) == sizeof(std::uint32_t), int> = 0>
constexpr _T
__sort_identity()
{
    if constexpr (__is_ascending)
        return ::sycl::bit_cast<_T>(0x7FFF'FFFFu);
    else
        return ::sycl::bit_cast<_T>(0xFFFF'FFFFu);
}

template <typename _T, bool __is_ascending,
          std::enable_if_t<std::is_floating_point_v<_T> && sizeof(_T) == sizeof(std::uint64_t), int> = 0>
constexpr _T
__sort_identity()
{
    if constexpr (__is_ascending)
        return ::sycl::bit_cast<_T>(0x7FFF'FFFF'FFFF'FFFFu);
    else
        return ::sycl::bit_cast<_T>(0xFFFF'FFFF'FFFF'FFFFu);
}

//-----------------------------------------------------------------------------
// Order-preserving casts: transform types so bit patterns sort correctly
//-----------------------------------------------------------------------------

// Unsigned integers: identity for ascending, bitwise NOT for descending
template <bool __is_ascending, typename _UInt, std::enable_if_t<std::is_unsigned_v<_UInt>, int> = 0>
inline _UInt
__order_preserving_cast(_UInt __val)
{
    if constexpr (__is_ascending)
        return __val;
    else
        return ~__val;
}

// Signed integers: XOR with sign bit
template <bool __is_ascending, typename _Int,
          std::enable_if_t<std::is_integral_v<_Int> && std::is_signed_v<_Int>, int> = 0>
inline std::make_unsigned_t<_Int>
__order_preserving_cast(_Int __val)
{
    using _UInt = std::make_unsigned_t<_Int>;
    // Flip sign bit for ascending, flip all bits for descending
    constexpr _UInt __mask =
        (__is_ascending) ? _UInt(1) << std::numeric_limits<_Int>::digits
                         : std::numeric_limits<_UInt>::max() >> 1;
    return ::sycl::bit_cast<_UInt>(__val) ^ __mask;
}

// Floating point (32-bit): handle sign and exponent correctly
template <bool __is_ascending, typename _Float,
          std::enable_if_t<std::is_floating_point_v<_Float> && sizeof(_Float) == sizeof(std::uint32_t), int> = 0>
inline std::uint32_t
__order_preserving_cast(_Float __val)
{
    std::uint32_t __uint_val = ::sycl::bit_cast<std::uint32_t>(__val);
    bool __is_negative = (__uint_val >> 31) != 0;

    std::uint32_t __mask;
    if constexpr (__is_ascending)
        __mask = __is_negative ? 0xFFFFFFFFu : 0x80000000u;
    else
        __mask = __is_negative ? 0x00000000u : 0x7FFFFFFFu;

    return __uint_val ^ __mask;
}

// Floating point (64-bit)
template <bool __is_ascending, typename _Float,
          std::enable_if_t<std::is_floating_point_v<_Float> && sizeof(_Float) == sizeof(std::uint64_t), int> = 0>
inline std::uint64_t
__order_preserving_cast(_Float __val)
{
    std::uint64_t __uint_val = ::sycl::bit_cast<std::uint64_t>(__val);
    bool __is_negative = (__uint_val >> 63) != 0;

    std::uint64_t __mask;
    if constexpr (__is_ascending)
        __mask = __is_negative ? 0xFFFFFFFFFFFFFFFFu : 0x8000000000000000u;
    else
        __mask = __is_negative ? 0x0000000000000000u : 0x7FFFFFFFFFFFFFFFu;

    return __uint_val ^ __mask;
}

//-----------------------------------------------------------------------------
// Extract radix bits from a key
//-----------------------------------------------------------------------------
template <std::uint16_t __radix_mask, typename _T, std::enable_if_t<std::is_unsigned_v<_T>, int> = 0>
inline std::uint16_t
__get_bucket(_T __value, std::uint32_t __radix_offset)
{
    return static_cast<std::uint16_t>((__value >> __radix_offset) & __radix_mask);
}

//-----------------------------------------------------------------------------
// Range pack: Unified interface for key-only and key-value sorting
//-----------------------------------------------------------------------------

// Dummy type for key-only sorting
struct __rng_dummy
{
};

// Helper to extract value type from range
template <typename _Rng>
struct __rng_value_type_deducer
{
    using __value_t = typename _Rng::value_type;
};

template <>
struct __rng_value_type_deducer<__rng_dummy>
{
    using __value_t = void;
};

// Range pack: holds keys and optionally values
template <typename _Rng1, typename _Rng2 = __rng_dummy>
struct __rng_pack
{
    using _KeyT = typename __rng_value_type_deducer<_Rng1>::__value_t;
    using _ValT = typename __rng_value_type_deducer<_Rng2>::__value_t;
    static constexpr bool __has_values = !std::is_void_v<_ValT>;

    const auto&
    __keys_rng() const
    {
        return __m_keys_rng;
    }

    const auto&
    __vals_rng() const
    {
        static_assert(__has_values);
        return __m_vals_rng;
    }

    __rng_pack(const _Rng1& __rng1, const _Rng2& __rng2 = __rng_dummy{})
        : __m_keys_rng(__rng1), __m_vals_rng(__rng2)
    {
    }

    __rng_pack(_Rng1&& __rng1, _Rng2&& __rng2 = __rng_dummy{})
        : __m_keys_rng(std::move(__rng1)), __m_vals_rng(std::move(__rng2))
    {
    }

  private:
    _Rng1 __m_keys_rng;
    _Rng2 __m_vals_rng;
};

//-----------------------------------------------------------------------------
// Helper to access underlying data from range
//-----------------------------------------------------------------------------
template <typename _Rng>
auto
__rng_data(const _Rng& __rng)
{
    return __rng.begin();
}

// Specialization for all_view (SYCL accessor)
// NOTE: all_view signature depends on whether PR #2551 is merged
template <typename _T, ::sycl::access::mode _M, __dpl_sycl::__target _Target, ::sycl::access::placeholder _Placeholder>
auto
__rng_data(const oneapi::dpl::__ranges::all_view<_T, _M, _Target, _Placeholder>& __view)
{
    return __view.accessor();
}

} // namespace oneapi::dpl::experimental::kt::gpu::__sycl::__impl

#endif // _ONEDPL_KT_SYCL_RADIX_SORT_UTILS_H
