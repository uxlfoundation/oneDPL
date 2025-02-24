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
template <typename _T1, typename = void>
struct __evenly_divided_binhash;

template <typename _T1>
struct __evenly_divided_binhash<_T1, ::std::enable_if_t<::std::is_floating_point_v<_T1>>>
{
    _T1 __minimum;
    _T1 __maximum;
    _T1 __scale;

    __evenly_divided_binhash(const _T1& __min, const _T1& __max, ::std::size_t __num_bins)
        : __minimum(__min), __maximum(__max), __scale(_T1(__num_bins) / (__max - __min))
    {
        assert(__num_bins < ::std::numeric_limits<::std::int32_t>::max());
    }

    template <typename _T2>
    ::std::int32_t
    get_bin(_T2 __value) const
    {
        ::std::int32_t ret = -1;
        if ((__value >= __minimum) && (__value < __maximum))
        {
            ret = (__value - __minimum) * __scale;
        }
        return ret;
    }
};

template <typename _T1>
struct __evenly_divided_binhash<_T1, ::std::enable_if_t<!::std::is_floating_point_v<_T1>>>
{
    _T1 __minimum;
    _T1 __range_size;
    ::std::int32_t __num_bins;
    __evenly_divided_binhash(const _T1& __min, const _T1& __max, ::std::size_t __num_bins_)
        : __minimum(__min), __range_size(__max - __min), __num_bins(__num_bins_)
    {
        assert(__num_bins < ::std::numeric_limits<::std::int32_t>::max());
    }

    template <typename _T2>
    ::std::int32_t
    get_bin(_T2 __value) const
    {
        ::std::int32_t ret = -1;
        if ((__value >= __minimum) && (__value < (__minimum + __range_size)))
        {
            ret = ::std::uint64_t(__value - __minimum) * ::std::uint64_t(__num_bins) / __range_size;
        }
        return ret;
    }
};

template <typename _Acc, typename _T2, typename _T3>
::std::int32_t
__custom_boundary_get_bin_helper(_Acc __acc, ::std::int32_t __size, _T2 __value, _T3 __min, _T3 __max)
{
    ::std::int32_t ret = -1;
    if (__value >= __min && __value < __max)
    {
        ret =
            oneapi::dpl::__internal::__pstl_upper_bound(__acc, ::std::int32_t{0}, __size, __value, ::std::less<_T2>{}) -
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
        assert(std::distance(__boundary_first, __boundary_last) < std::numeric_limits<std::int32_t>::max());
    }

    template <typename _T2>
    auto
    get_bin(_T2 __value) const
    {
        auto __size = ::std::distance(__boundary_first, __boundary_last);
        return __custom_boundary_get_bin_helper(__boundary_first, __size, __value, __boundary_first[0],
                                                __boundary_first[__size - 1]);
    }
};

} // end namespace __internal
} // end namespace dpl
} // end namespace oneapi

#endif // _ONEDPL_HISTOGRAM_BINHASH_UTILS_H
