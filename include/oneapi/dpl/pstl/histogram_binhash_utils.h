// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// This file incorporates work covered by the following copyright and permission
// notice:
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#ifndef _ONEDPL_HISTOGRAM_BINHASH_UTILS_H
#define _ONEDPL_HISTOGRAM_BINHASH_UTILS_H
#include <algorithm>
#include <utility>
#include <iterator>
#include <cstdint>
#include <type_traits>
#include <cassert>
#include <limits>

namespace oneapi
{
namespace dpl
{
namespace __internal
{

using __bin_idx_t = std::int32_t;

template <typename _T1>
struct __evenly_divided_binhash_fp
{
    _T1 __minimum;
    _T1 __maximum;
    _T1 __scale;
    __bin_idx_t __num_bins;

    __evenly_divided_binhash_fp(const _T1& __min, const _T1& __max, std::size_t __num_bins_)
        : __minimum(__min), __maximum(__max), __scale(__max > __min ? _T1(__num_bins_) / (__max - __min) : _T1(0)),
          __num_bins(__num_bins_)
    {
        assert(__num_bins_ < std::numeric_limits<__bin_idx_t>::max());
    }

    template <typename _T2>
    __bin_idx_t
    get_bin(_T2 __value) const
    {
        __bin_idx_t ret = -1;
        if ((__value >= __minimum) && (__value < __maximum))
        {
            // The rounded scale can carry a sample just below the maximum to num_bins
            ret = std::clamp<__bin_idx_t>((__value - __minimum) * __scale, 0, __num_bins - 1);
        }
        return ret;
    }
};

// The bin of an integer sample is floor(offset * num_bins / range). For bound types of at most 32 bits the product
// fits in 64 bits, so a division is exact. For 64-bit bounds it can overflow, and ranges where it may are handled by
// storing num_bins / range as a 128-bit fixed-point fraction instead.
template <typename _T1, typename _RangeT>
struct __evenly_divided_binhash_int_base
{
    template <typename _T>
    static constexpr bool __is_supported_int_v = std::is_integral_v<_T> && sizeof(_T) <= sizeof(std::uint64_t);

    static_assert(__is_supported_int_v<_T1>, "histogram bounds must be integers of at most 64 bits, or floating point");

    _T1 __minimum;
    _T1 __maximum;
    _RangeT __range; // max - min
    __bin_idx_t __num_bins;

    // The distance from __lower up to __upper, where __lower <= __upper in their common type C, which the bounds
    // checks compare in. Subtracting in unsigned C is exact even where it overflows C, since the true distance fits
    // in its width.
    template <typename _Lower, typename _Upper>
    static std::make_unsigned_t<std::enable_if_t<std::is_integral_v<_Lower> && std::is_integral_v<_Upper>,
                                std::common_type_t<_Lower, _Upper>>>
    __unsigned_distance(_Lower __lower, _Upper __upper)
    {
        using _UnsignedT = std::make_unsigned_t<std::common_type_t<_Lower, _Upper>>;
        return _UnsignedT(__upper) - _UnsignedT(__lower);
    }

    __evenly_divided_binhash_int_base(const _T1& __min, const _T1& __max, std::size_t __num_bins_)
        : __minimum(__min), __maximum(__max), __range(_RangeT(__unsigned_distance(__min, __max))),
          __num_bins(__num_bins_)
    {
        assert(__num_bins_ < std::numeric_limits<__bin_idx_t>::max());
    }

    // Exact only where offset * num_bins fits in 64 bits
    template <typename _T2>
    __bin_idx_t
    __get_bin_by_division(_T2 __value) const
    {
        static_assert(__is_supported_int_v<_T2>,
                      "histogram samples must be integers of at most 64 bits, or floating point");
        __bin_idx_t ret = -1;
        if ((__value >= __minimum) && (__value < __maximum))
            ret = __unsigned_distance(__minimum, __value) * std::uint64_t(__num_bins) / __range;
        return ret;
    }
};

// Bounds of at most 32 bits, where the division is always exact
template <typename _T1>
struct __evenly_divided_binhash_cannot_overflow : __evenly_divided_binhash_int_base<_T1, std::uint32_t>
{
    using __evenly_divided_binhash_int_base<_T1, std::uint32_t>::__evenly_divided_binhash_int_base;

    template <typename _T2>
    __bin_idx_t
    get_bin(_T2 __value) const
    {
        return this->__get_bin_by_division(__value);
    }
};

// 64-bit bounds, where ranges too large for the division use the fixed-point fraction
template <typename _T1>
struct __evenly_divided_binhash_can_overflow : __evenly_divided_binhash_int_base<_T1, std::uint64_t>
{
    using _Base = __evenly_divided_binhash_int_base<_T1, std::uint64_t>;

    bool __large_range; // offset * num_bins may not fit in 64 bits
    // floor(num_bins * 2^128 / range) as its high and low words, used only for large ranges
    std::uint64_t __frac;
    std::uint64_t __frac2;

    __evenly_divided_binhash_can_overflow(const _T1& __min, const _T1& __max, std::size_t __num_bins_)
        : _Base(__min, __max, __num_bins_), __large_range(false), __frac(0), __frac2(0)
    {
        __large_range = __num_bins_ != 0 && this->__range > std::numeric_limits<std::uint64_t>::max() / __num_bins_;
        if (!__large_range)
            return;
        // Long division of num_bins * 2^128 by range, shifting quotient bits in through the low word. A large range
        // exceeds UINT64_MAX / num_bins >= 2^33, so num_bins < range and the remainder stays below range throughout
        std::uint64_t __rem = std::uint64_t(this->__num_bins);
        for (int __i = 0; __i < 128; ++__i)
        {
            const bool __carry = __rem >> 63;
            __rem <<= 1;
            __frac = (__frac << 1) | (__frac2 >> 63);
            __frac2 <<= 1;
            if (__carry || __rem >= this->__range)
            {
                __rem -= this->__range;
                __frac2 |= 1;
            }
        }
    }

    // Valid for any range, but only needed for large ones
    template <typename _T2>
    __bin_idx_t
    __get_bin_by_fraction(_T2 __value) const
    {
        static_assert(_Base::template __is_supported_int_v<_T2>,
                      "histogram samples must be integers of at most 64 bits, or floating point");
        __bin_idx_t ret = -1;
        if ((__value >= this->__minimum) && (__value < this->__maximum))
        {
            const std::uint64_t __offset = _Base::__unsigned_distance(this->__minimum, __value);
            // The correct bin or the one below it, due to truncation of the fixed-point multiplication
            const std::uint64_t __bin = __dpl_mul_hi_u64(__offset, __frac);
            // 2^64 * (offset * num_bins / range - bin) is lo(offset * frac) + hi(offset * frac2) plus truncated terms
            // in [0, 2). The bin steps up when that reaches 2^64, which as range < 2^64 happens exactly when the
            // computed sum reaches 2^64 - 1, i.e. lo(offset * frac) >= ~hi(offset * frac2)
            ret = __bin + (__offset * __frac >= ~__dpl_mul_hi_u64(__offset, __frac2));
        }
        return ret;
    }

    template <typename _T2>
    __bin_idx_t
    get_bin(_T2 __value) const
    {
        return __large_range ? __get_bin_by_fraction(__value) : this->__get_bin_by_division(__value);
    }
};

// Creates the evenly divided binhash suited to the bound type
template <typename _T1>
auto
__make_evenly_divided_binhash(const _T1& __min, const _T1& __max, std::size_t __num_bins)
{
    if constexpr (std::is_floating_point_v<_T1>)
        return __evenly_divided_binhash_fp<_T1>(__min, __max, __num_bins);
    else if constexpr (sizeof(_T1) > sizeof(std::uint32_t))
        return __evenly_divided_binhash_can_overflow<_T1>(__min, __max, __num_bins);
    else
        return __evenly_divided_binhash_cannot_overflow<_T1>(__min, __max, __num_bins);
}

template <typename _Acc, typename _T2, typename _T3>
__bin_idx_t
__custom_boundary_get_bin_helper(_Acc __acc, __bin_idx_t __size, _T2 __value, _T3 __min, _T3 __max)
{
    __bin_idx_t ret = -1;
    if (__value >= __min && __value < __max)
    {
        _T2* __value_ptr = &__value;
        ret = oneapi::dpl::__internal::__pstl_upper_bound(__acc, __bin_idx_t{0}, __size, __value_ptr, std::less<_T2>{},
                                                          oneapi::dpl::identity{}, oneapi::dpl::identity{}) -
              1;
    }
    return ret;
}

template <typename _RandomAccessIterator>
struct __custom_boundary_binhash
{
    _RandomAccessIterator __boundary_first;
    _RandomAccessIterator __boundary_last;
    __custom_boundary_binhash(_RandomAccessIterator __boundary_first_, _RandomAccessIterator __boundary_last_)
        : __boundary_first(__boundary_first_), __boundary_last(__boundary_last_)
    {
        assert(std::distance(__boundary_first, __boundary_last) < std::numeric_limits<__bin_idx_t>::max());
    }

    template <typename _T2>
    __bin_idx_t
    get_bin(_T2 __value) const
    {
        __bin_idx_t __size = ::std::distance(__boundary_first, __boundary_last);
        return __custom_boundary_get_bin_helper(__boundary_first, __size, __value, __boundary_first[0],
                                                __boundary_first[__size - 1]);
    }
};

} // end namespace __internal
} // end namespace dpl
} // end namespace oneapi

#endif // _ONEDPL_HISTOGRAM_BINHASH_UTILS_H
