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

#ifndef _HISTOGRAM_SERIAL_IMPL_H
#define _HISTOGRAM_SERIAL_IMPL_H

template <typename T1, typename T2, typename Size>
::std::enable_if_t<!std::is_floating_point_v<T1>, ::std::uint32_t>
get_bin(T1 value, T2 min, T2 max, Size num_bins)
{
    return ::std::uint32_t((::std::uint64_t(value - min) * ::std::uint64_t(num_bins)) / (max - min));
}

template <typename T1, typename T2, typename Size>
::std::enable_if_t<std::is_floating_point_v<T1>, ::std::uint32_t>
get_bin(T1 value, T2 min, T2 max, Size num_bins)
{
    return ::std::uint32_t((value - min) * (T1(num_bins) / (max - min)));
}

template <typename _InputIterator1, typename _Size, typename _T, typename _OutputIterator>
_OutputIterator
histogram_sequential(_InputIterator1 __first, _InputIterator1 __last, _Size __num_bins, _T __first_bin_min_val,
                     _T __last_bin_max_val, _OutputIterator __histogram_first)
{
    ::std::fill_n(__histogram_first, __num_bins, 0);

    for (auto __tmp = __first; __tmp < __last; ++__tmp)
    {
        auto __value = *__tmp;
        if (__value >= __first_bin_min_val && __value < __last_bin_max_val)
        {
            _Size __bin = get_bin(__value, __first_bin_min_val, __last_bin_max_val, __num_bins);
            ++(__histogram_first[__bin]);
        }
    }
    return __histogram_first + __num_bins;
}

template <typename _InputIterator1, typename _InputIterator2, typename _OutputIterator>
_OutputIterator
histogram_sequential(_InputIterator1 __first, _InputIterator1 __last, _InputIterator2 __boundary_first,
                     _InputIterator2 __boundary_last, _OutputIterator __histogram_first)
{
    int __num_bins = (__boundary_last - __boundary_first) - 1;
    ::std::fill_n(__histogram_first, __num_bins, 0);

    for (auto __tmp = __first; __tmp < __last; ++__tmp)
    {
        auto __value = *__tmp;
        if ((__value >= (*__boundary_first)) && (__value < (*(__boundary_last - 1))))
        {
            ::std::ptrdiff_t bin =
                (::std::upper_bound(__boundary_first, __boundary_last, __value) - __boundary_first) - 1;
            ++(__histogram_first[bin]);
        }
    }
    return __histogram_first + __num_bins;
}

#endif // _HISTOGRAM_SERIAL_IMPL_H
