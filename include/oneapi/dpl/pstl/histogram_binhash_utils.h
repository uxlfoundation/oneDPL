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

template <typename _T1, typename = void>
struct __evenly_divided_binhash;

// A view of a binhash whose bin regime is fixed at compile time, so a loop of get_bin calls carries no regime branch
template <typename _BinHash, bool _LargeRange>
struct __fixed_regime_binhash
{
    const _BinHash& __bin_hash;

    template <typename _T2>
    __bin_idx_t
    get_bin(_T2 __value) const
    {
        return __bin_hash.template __get_bin_in_regime<_LargeRange>(__value);
    }
};

template <typename _T1>
struct __evenly_divided_binhash<_T1, std::enable_if_t<std::is_floating_point_v<_T1>>>
{
    _T1 __minimum;
    _T1 __maximum;
    _T1 __scale;
    __bin_idx_t __num_bins;

    __evenly_divided_binhash(const _T1& __min, const _T1& __max, std::size_t __num_bins_)
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

template <typename _T1>
struct __evenly_divided_binhash<_T1, std::enable_if_t<!std::is_floating_point_v<_T1>>>
{
    // The bin of a sample is floor(offset * num_bins / range). For bound types of at most 32 bits the product fits
    // in 64 bits, so a division is exact. For 64-bit bounds it can overflow, and ranges where it may are handled by
    // storing num_bins / range as a 64-bit fixed-point fraction instead. Multiplying the offset by it truncates to the
    //correct bin or the one below, which one exact 128-bit comparison corrects. The division is faster on devices
    //without native 64-bit integer multiplication, so it serves every range where it is exact.
    static constexpr bool __may_overflow = sizeof(_T1) > sizeof(std::uint32_t);
    // The range fits in an unsigned type the size of _T1; keeping it narrow keeps the division narrow
    using _RangeT = std::conditional_t<__may_overflow, std::uint64_t, std::uint32_t>;

    _T1 __minimum;
    _T1 __maximum;
    _RangeT __range; // max - min
    __bin_idx_t __num_bins;
    bool __large_range;   // offset * num_bins may not fit in 64 bits
    std::uint64_t __frac; // floor(num_bins * 2^64 / range), used only for large ranges

    // The distance from __lo up to __hi, where __lo <= __hi in their common type C, which the bounds checks compare
    // in. Subtracting in unsigned C is exact even where it overflows C, since the true distance fits in its width.
    template <typename _Lo, typename _Hi>
    static std::uint64_t
    __unsigned_distance(_Lo __lo, _Hi __hi)
    {
        using _CommonT = decltype(__hi - __lo);
        static_assert(std::is_integral_v<_CommonT> && sizeof(_CommonT) <= sizeof(std::uint64_t),
                      "histogram bounds and samples must be integers of at most 64 bits, or floating point");
        using _UnsignedT = std::make_unsigned_t<_CommonT>;
        return _UnsignedT(__hi) - _UnsignedT(__lo);
    }

    __evenly_divided_binhash(const _T1& __min, const _T1& __max, std::size_t __num_bins_)
        : __minimum(__min), __maximum(__max), __range(_RangeT(__unsigned_distance(__min, __max))),
          __num_bins(__num_bins_), __large_range(false), __frac(0)
    {
        assert(__num_bins_ < std::numeric_limits<__bin_idx_t>::max());
        if constexpr (__may_overflow)
        {
            __large_range = __num_bins_ != 0 && __range > std::numeric_limits<std::uint64_t>::max() / __num_bins_;
            if (!__large_range)
                return;
            // Long division of num_bins * 2^64 by range. A large range exceeds UINT64_MAX / num_bins >= 2^33, so
            // num_bins < range and the remainder stays below range throughout
            std::uint64_t __rem = std::uint64_t(__num_bins);
            for (int __i = 0; __i < 64; ++__i)
            {
                const bool __carry = __rem >> 63;
                __rem <<= 1;
                __frac <<= 1;
                if (__carry || __rem >= __range)
                {
                    __rem -= __range;
                    __frac |= 1;
                }
            }
        }
    }

    std::uint64_t
    __large_range_bin(std::uint64_t __offset) const
    {
        const std::uint64_t __bins = __num_bins;
        // The correct bin or the one below it
        std::uint64_t __bin = __dpl_mul_hi_u64(__offset, __frac);
        // Step up when the next bin starts at or before the offset: (bin + 1) * range <= offset * num_bins,
        // compared as 128-bit products
        const std::uint64_t __next_start_hi = __dpl_mul_hi_u64(__bin + 1, __range);
        const std::uint64_t __next_start_lo = (__bin + 1) * __range;
        const std::uint64_t __scaled_hi = __dpl_mul_hi_u64(__offset, __bins);
        const std::uint64_t __scaled_lo = __offset * __bins;
        return __bin +
               (__next_start_hi < __scaled_hi || (__next_start_hi == __scaled_hi && __next_start_lo <= __scaled_lo));
    }

    template <bool _LargeRange, typename _T2>
    __bin_idx_t
    __get_bin_in_regime(_T2 __value) const
    {
        __bin_idx_t ret = -1;
        if ((__value >= __minimum) && (__value < __maximum))
        {
            const std::uint64_t __offset = __unsigned_distance(__minimum, __value);
            if constexpr (_LargeRange)
                ret = __large_range_bin(__offset);
            else
                ret = __offset * std::uint64_t(__num_bins) / __range;
        }
        return ret;
    }

    template <typename _T2>
    __bin_idx_t
    get_bin(_T2 __value) const
    {
        if constexpr (__may_overflow)
        {
            if (__large_range)
                return __get_bin_in_regime<true>(__value);
        }
        return __get_bin_in_regime<false>(__value);
    }

    // Calls __f with a binhash whose regime is fixed at compile time, so callers binning many samples can test
    // __large_range once instead of per sample
    static constexpr bool __has_bin_regimes = __may_overflow;

    // called by free function __with_bin_regime to visit the bin regime of the wrapped binhash
    template <typename _F>
    void
    __visit_bin_regime(_F&& __f) const
    {
        if (__large_range)
            __f(__fixed_regime_binhash<__evenly_divided_binhash, true>{*this});
        else
            __f(__fixed_regime_binhash<__evenly_divided_binhash, false>{*this});
    }
};

template <typename _BinHash, typename = void>
struct __has_bin_regimes : std::false_type
{
};

template <typename _BinHash>
struct __has_bin_regimes<_BinHash, std::enable_if_t<_BinHash::__has_bin_regimes>> : std::true_type
{
};

// Calls __f with __bin_hash, or with a view of it whose bin regime is fixed, for binhashes that have regimes
template <typename _BinHash, typename _F>
void
__with_bin_regime(const _BinHash& __bin_hash, _F&& __f)
{
    if constexpr (__has_bin_regimes<_BinHash>::value)
        __bin_hash.__visit_bin_regime(std::forward<_F>(__f));
    else
        __f(__bin_hash);
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
